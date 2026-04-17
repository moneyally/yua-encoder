"""emotion-project / build_soft_labels.py — annot_A/B/C vote → soft target.

===============================================================================
설계 (DESIGN)
===============================================================================

목적
----
annot_A/B/C 3인의 `faceExp` (한국어 감정 라벨) 를 우리 4-class 체계로 매핑 후
vote count → soft target (길이 4, sum=1) 생성. Stage-0 (build) 는 학습과 분리된
오프라인 단계. finetune_soft.py 가 이 npz 를 읽어 KL-div 미세조정에 사용.

입력
----
- data_rot/label/{train,val}/<split>_<cls>.json  (encoding: euc-kr → utf-8 fallback)
- scripts/emotion_mapping.json  (선택; 없으면 내장 기본 사용)

출력
----
- results/soft_labels/<split>_soft.npz  (keys: filenames, soft_targets, class_idx_gt,
                                          coverage, classes)

매핑 규칙 (내장 기본)
--------------------
anger   ← {"분노","짜증","화남","격분"}
happy   ← {"기쁨","행복","즐거움","만족"}
panic   ← {"놀람","공포","불안","당황","두려움"}
sadness ← {"슬픔","우울","상처","허탈"}

매핑 외 값 (예: "중립"/"수치심" 등) → 해당 라벨러 표 1개 skip.
3인 모두 매핑 실패 시 → fallback: 폴더 GT one-hot 에 label_smoothing 적용.

수식 (soft target)
------------------
votes = [c for c in {A,B,C} if mapping(c) != None]
if len(votes) == 0:
    fallback: soft = onehot(gt) with smoothing s  → [s/3, s/3, s/3, s/3] with gt idx = 1-s
else:
    counts[c] = #{i : votes[i] == c}
    soft[c]   = counts[c] / sum(counts)

optional: gt_mix (0~1) 로 one-hot(gt) 와 혼합 — vote overload 방어
         soft_final = (1 - gt_mix) * soft + gt_mix * onehot(gt)

optional: label_smoothing ε → soft_final = (1-ε)*soft_final + ε/K  (K=4)

assertion: abs(sum(soft_final) - 1.0) < 1e-6  (매 샘플 sanity check)

CLI
---
--data-root (default: $EMOTION_PROJECT_ROOT/data_rot)
--out-dir   (default: $EMOTION_PROJECT_ROOT/results/soft_labels)
--mapping-json (default: scripts/emotion_mapping.json if exists, else builtin)
--gt-mix (float, 0.0~1.0, default 0.0)
--label-smoothing (float, 0.0~0.3, default 0.05)
--splits train val  (space-separated)
--min-raters (int, default 1) — vote 가 이 수 미만이면 fallback 처리
--verbose

산출물 통계 (stdout)
-------------------
- total / mapped(3·2·1·0) 카운트
- soft argmax vs GT mismatch rate
- soft target entropy 평균 (다양성 지표)
===============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 상수 / 기본 매핑
# --------------------------------------------------------------------------
PROJECT = Path(os.environ.get(
    "EMOTION_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
CLASSES = ["anger", "happy", "panic", "sadness"]   # train_vit.py 와 동일 순서 고정
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# 내장 기본 매핑 (한국어 faceExp → 4-class key)
BUILTIN_MAPPING: dict[str, str] = {
    # anger
    "분노": "anger", "짜증": "anger", "화남": "anger", "격분": "anger",
    # happy
    "기쁨": "happy", "행복": "happy", "즐거움": "happy", "만족": "happy",
    # panic
    "놀람": "panic", "공포": "panic", "불안": "panic",
    "당황": "panic", "두려움": "panic",
    # sadness
    "슬픔": "sadness", "우울": "sadness", "상처": "sadness", "허탈": "sadness",
}


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def load_json_eucmix(path: Path) -> list:
    """euc-kr 우선 → 실패 시 utf-8. (train_vit.py 규칙)"""
    for enc in ("euc-kr", "utf-8"):
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        except FileNotFoundError:
            return []
    return []


def load_mapping(args) -> dict[str, str]:
    """--mapping-json 있으면 거기서, 없으면 builtin.
    mapping file 포맷: {"분노": "anger", ...} 단순 dict.
    """
    mp_path = args.mapping_json
    if mp_path is None:
        # 기본 경로 탐색
        default_path = PROJECT / "scripts" / "emotion_mapping.json"
        if default_path.is_file():
            mp_path = default_path
    if mp_path is not None:
        p = Path(mp_path)
        if not p.is_absolute():
            # 상대경로는 프로젝트 루트 기준 (CWD 의존 제거)
            p = PROJECT / p
        if not p.is_file():
            print(f"[!] mapping-json not found: {p}  → builtin 사용")
            return dict(BUILTIN_MAPPING)
        try:
            with open(p, encoding="utf-8") as f:
                user_map = json.load(f)
            if not isinstance(user_map, dict):
                raise ValueError("mapping JSON 은 dict 여야 함")
            # value 는 CLASSES 안의 값이어야 함
            for k, v in user_map.items():
                if v not in CLASS_TO_IDX:
                    raise ValueError(
                        f"mapping value '{v}' (key='{k}') 는 {CLASSES} 에 없음"
                    )
            return {str(k): str(v) for k, v in user_map.items()}
        except Exception as e:
            print(f"[!] mapping-json 로드 실패: {e}  → builtin fallback")
            return dict(BUILTIN_MAPPING)
    return dict(BUILTIN_MAPPING)


# --------------------------------------------------------------------------
# 핵심 로직
# --------------------------------------------------------------------------
def faceexp_to_idx(value: str | None, mapping: dict[str, str]) -> int | None:
    """단일 faceExp → class_idx (매핑 실패 시 None)."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    cls = mapping.get(v)
    if cls is None:
        return None
    return CLASS_TO_IDX.get(cls)


def compute_soft(item: dict,
                 gt_idx: int,
                 mapping: dict[str, str],
                 gt_mix: float,
                 label_smoothing: float,
                 min_raters: int) -> tuple[np.ndarray, str]:
    """한 샘플 item → (soft_target[4], coverage_str).
    coverage_str 예: "3/3", "2/3", "1/3", "0/3_fallback"
    """
    votes: list[int] = []
    for rater_key in ("annot_A", "annot_B", "annot_C"):
        rater = item.get(rater_key)
        if not isinstance(rater, dict):
            continue
        idx = faceexp_to_idx(rater.get("faceExp"), mapping)
        if idx is not None:
            votes.append(idx)

    n_votes = len(votes)

    # vote 수 부족 → fallback one-hot(gt) + smoothing. CLI 에서 min_raters∈[1,3] 보장.
    if n_votes < min_raters:
        base = np.zeros(NUM_CLASSES, dtype=np.float64)
        base[gt_idx] = 1.0
        coverage = f"{n_votes}/3_fallback"
    else:
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for v in votes:
            counts[v] += 1.0
        s = counts.sum()
        if s <= 0:  # 이론상 도달 불가, 방어
            base = np.zeros(NUM_CLASSES, dtype=np.float64)
            base[gt_idx] = 1.0
            coverage = f"{n_votes}/3_fallback_zero"
        else:
            base = counts / s
            coverage = f"{n_votes}/3"

    # gt_mix (0~1): one-hot(gt) 와 혼합
    if gt_mix > 0:
        onehot = np.zeros(NUM_CLASSES, dtype=np.float64)
        onehot[gt_idx] = 1.0
        base = (1.0 - gt_mix) * base + gt_mix * onehot

    # label_smoothing: uniform 과 혼합
    if label_smoothing > 0:
        uniform = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float64)
        base = (1.0 - label_smoothing) * base + label_smoothing * uniform

    # NaN 방어 + renormalize (부동소수 오차)
    if not np.all(np.isfinite(base)):
        base = np.zeros(NUM_CLASSES, dtype=np.float64)
        base[gt_idx] = 1.0
        coverage += "_nanfix"
    s = base.sum()
    if s <= 0 or not math.isfinite(s):
        base = np.zeros(NUM_CLASSES, dtype=np.float64)
        base[gt_idx] = 1.0
    else:
        base = base / s

    # sanity
    total = float(base.sum())
    assert abs(total - 1.0) < 1e-6, (
        f"soft sum != 1 (got {total:.9f})  votes={votes}  gt={gt_idx}  "
        f"gt_mix={gt_mix}  smooth={label_smoothing}"
    )
    return base.astype(np.float32), coverage


def process_split(data_root: Path,
                  split: str,
                  mapping: dict[str, str],
                  args) -> tuple[list[str], np.ndarray, np.ndarray, list[str], dict]:
    """한 split 전체 처리. 반환:
        filenames (N,), soft (N,4), gt_idx (N,), coverage (N,), stats dict
    """
    lbl_dir = data_root / "label" / split
    if not lbl_dir.is_dir():
        raise SystemExit(f"[!] label 디렉토리 없음: {lbl_dir}")

    filenames: list[str] = []
    soft_list: list[np.ndarray] = []
    gt_list: list[int] = []
    cov_list: list[str] = []

    cov_counter: Counter = Counter()
    argmax_match = 0
    entropy_sum = 0.0

    for cls_idx, cls in enumerate(CLASSES):
        p = lbl_dir / f"{split}_{cls}.json"
        data = load_json_eucmix(p)
        if not data:
            print(f"[warn] {p} 비어있음 또는 파싱 실패")
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            fname = item.get("filename")
            if not fname or not isinstance(fname, str):
                continue
            soft, cov = compute_soft(
                item, cls_idx, mapping,
                gt_mix=args.gt_mix,
                label_smoothing=args.label_smoothing,
                min_raters=args.min_raters,
            )
            filenames.append(fname)
            soft_list.append(soft)
            gt_list.append(cls_idx)
            cov_list.append(cov)

            cov_counter[cov] += 1
            if int(np.argmax(soft)) == cls_idx:
                argmax_match += 1
            # entropy = -Σ p log p  (0*log0 = 0 처리)
            p_nz = soft[soft > 1e-12]
            entropy_sum += float(-np.sum(p_nz * np.log(p_nz)))

    N = len(filenames)
    stats = {
        "split": split,
        "n": N,
        "coverage": dict(cov_counter),
        "argmax_match_gt": argmax_match,
        "argmax_match_rate": (argmax_match / N) if N else 0.0,
        "entropy_mean": (entropy_sum / N) if N else 0.0,
        "gt_mix": args.gt_mix,
        "label_smoothing": args.label_smoothing,
        "min_raters": args.min_raters,
    }
    soft_arr = np.stack(soft_list, axis=0).astype(np.float32) if soft_list else \
        np.zeros((0, NUM_CLASSES), dtype=np.float32)
    gt_arr = np.array(gt_list, dtype=np.int64)
    return filenames, soft_arr, gt_arr, cov_list, stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="annot_A/B/C faceExp vote → soft target npz builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-root", default=str(PROJECT / "data_rot"),
                    help="data_rot 루트 (label/{train,val}/*.json 포함)")
    ap.add_argument("--out-dir", default=str(PROJECT / "results" / "soft_labels"),
                    help="npz 저장 디렉토리")
    ap.add_argument("--mapping-json", default=None,
                    help="JSON dict (한국어 faceExp → anger/happy/panic/sadness)")
    ap.add_argument("--gt-mix", type=float, default=0.0,
                    help="soft_final = (1-m)*vote + m*one_hot(gt)  (0~1)")
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="final uniform smoothing  (0~0.3 권장)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="처리할 split 목록")
    ap.add_argument("--min-raters", type=int, default=1,
                    help="vote 가 이 수 미만이면 fallback(GT one-hot) 처리. 범위 [1,3].")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # 경계값 체크
    if not (0.0 <= args.gt_mix <= 1.0):
        raise SystemExit(f"[!] --gt-mix 는 [0,1] 이어야 함 (입력: {args.gt_mix})")
    if not (0.0 <= args.label_smoothing <= 0.3):
        raise SystemExit(
            f"[!] --label-smoothing 은 [0,0.3] 이어야 함 (입력: {args.label_smoothing})"
        )
    if args.min_raters < 1 or args.min_raters > 3:
        raise SystemExit(f"[!] --min-raters 는 [1,3] 이어야 함 (입력: {args.min_raters})")
    return args


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args)
    if args.verbose:
        print(f"[i] mapping keys: {sorted(mapping.keys())}")
        print(f"[i] gt_mix={args.gt_mix}  label_smoothing={args.label_smoothing}  "
              f"min_raters={args.min_raters}")

    summary: dict = {"splits": {}, "mapping": mapping,
                     "classes": CLASSES}

    for split in args.splits:
        print(f"\n=== split: {split} ===")
        filenames, soft, gt, cov, stats = process_split(
            data_root, split, mapping, args,
        )
        if len(filenames) == 0:
            print(f"[!] {split}: 0 samples — skip 저장")
            continue

        out_path = out_dir / f"{split}_soft.npz"
        # dtype=str (고정폭 유니코드) 로 저장 → 로드 시 allow_pickle=False 로 안전
        np.savez(
            str(out_path),
            filenames=np.array(filenames, dtype=str),
            soft_targets=soft,
            class_idx_gt=gt,
            coverage=np.array(cov, dtype=str),
            classes=np.array(CLASSES, dtype=str),
        )
        print(f"[i] saved: {out_path}  (n={len(filenames)})")

        # 통계 출력
        print(f"    coverage: {stats['coverage']}")
        print(f"    argmax(soft)==gt : {stats['argmax_match_gt']} / {stats['n']} "
              f"({stats['argmax_match_rate']:.4f})")
        print(f"    entropy mean      : {stats['entropy_mean']:.4f}  "
              f"(uniform={math.log(NUM_CLASSES):.4f})")
        summary["splits"][split] = {**stats, "path": str(out_path)}

    # summary json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] summary → {summary_path}")


if __name__ == "__main__":
    main()
