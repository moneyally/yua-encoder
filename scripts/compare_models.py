"""emotion-project / scripts/compare_models.py

두 개(또는 그 이상) 학습된 모델을 val set 에서 **정밀 비교**.

================================================================================
목적
================================================================================

튜닝 전/후 모델 비교에 초점. 예: E5_original vs E7_soft_ft. 각 모델에 대해:

1. val_acc / val_macro_f1 / val_nll (overall)
2. per-class precision / recall / F1 (sklearn classification_report)
3. confusion matrix (각 모델)
4. 샘플 단위 delta — 한쪽만 맞춘 샘플 수, 둘 다 맞춤/오답 수
5. McNemar test (통계적 유의성) — chi-square continuity-corrected (statsmodels)
6. 결과 md + confusion matrix PNG 저장

================================================================================
설계 원칙 (CLAUDE.md #7 · 헌법)
================================================================================

- 하드코딩 금지 → argparse / EMOTION_PROJECT_ROOT env
- predict.py 의 load_model / predict_probs / CLASSES 재사용
- ensemble_search.py 의 collect_val_samples 재사용 (val + bbox 수집 로직 동일)
- bbox crop / raw 분기 — ensemble_search.py 와 동일 (학습 조건 일치)
- McNemar: (b + c) 합 0 방어, exact=False + correction=True (chi2 근사)
- NLL: np.clip(eps=1e-12) → log(0) 방지
- confusion_matrix(labels=list(range(4))) → 클래스 순서 고정
- stacking shape (N, 4) assertion

================================================================================
CLI 예시
================================================================================

python scripts/compare_models.py \\
  --models-a models/exp05_vit_b16_two_stage.pt \\
  --models-b models/exp07_vit_soft_ft.pt \\
  --name-a E5_original \\
  --name-b E7_soft_ft \\
  --val-dir data_rot/img/val \\
  --val-label-root data_rot/label \\
  --crop-mode bbox \\
  --output-dir results \\
  --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------
# 프로젝트 루트
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT = Path(os.environ.get("EMOTION_PROJECT_ROOT", str(THIS_FILE.parent.parent)))
# scripts/ 에 __init__.py 가 없어 `from scripts import ...` 방식은 불가능.
# scripts/ 자체와 PROJECT 둘 다 sys.path 에 올려 직접 import.
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

# predict.py / ensemble_search.py 재사용
import predict as predict_mod  # noqa: E402
import ensemble_search as es  # noqa: E402

CLASSES: List[str] = list(predict_mod.CLASSES)
NUM_CLASSES: int = predict_mod.NUM_CLASSES
assert NUM_CLASSES == 4 and CLASSES == ["anger", "happy", "panic", "sadness"], (
    f"predict.CLASSES 가 기대와 다름: {CLASSES}"
)


def _log(msg: str) -> None:
    print(f"[compare_models] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------
# probs 수집 (캐시 포함) — ensemble_search.collect_probs_for_model 재사용
# ------------------------------------------------------------------

def get_probs(
    model_path: Path,
    paths: List[Path],
    bboxes: List[Optional[Tuple[float, float, float, float]]],
    cache_dir: Path,
    crop_mode: str,
    force_cpu: bool,
    force_recompute: bool,
) -> np.ndarray:
    """(N, 4) softmax probabilities — crop_mode 에 따라 bbox crop 또는 raw.

    캐시 hit 시 재계산 생략. ensemble_search 와 동일한 캐시 키라서 두 스크립트간
    npz 공유 가능 (crop_mode 포함 → 별도 파일).
    """
    probs = es.collect_probs_for_model(
        model_path=model_path,
        paths=paths,
        cache_dir=cache_dir,
        force_cpu=force_cpu,
        force_recompute=force_recompute,
        bboxes=bboxes,
        crop_mode=crop_mode,
    )
    assert probs.shape == (len(paths), NUM_CLASSES), (
        f"probs shape 불일치: {probs.shape} (expected ({len(paths)},{NUM_CLASSES}))"
    )
    # 합계 1 재확인
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=5e-3):
        bad = int((np.abs(sums - 1.0) > 5e-3).sum())
        _log(f"[warn] {model_path.name} probs 합이 1 아닌 행 {bad}/{len(paths)} — renormalize")
        probs = probs / np.clip(sums[:, None], 1e-8, None)
    return probs.astype(np.float32)


# ------------------------------------------------------------------
# McNemar test
# ------------------------------------------------------------------

def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """binary correct 마스크 2개 → McNemar test (chi2 w/ continuity correction).

    2x2 contingency table:
      [[both_correct, only_a_correct],
       [only_b_correct, both_wrong]]

    off-diagonal b = only_a, c = only_b 에 대해
      χ² = (|b - c| - 1)² / (b + c)    (continuity correction)
    p-value = 1 - CDF_χ²(1) (χ²)

    b + c == 0 이면 p = 1.0 (두 모델 disagree 없음 → 차이 감지 불가).

    statsmodels 가 있으면 공식 구현 사용, 없으면 scipy.stats.chi2 로 직접 계산.
    """
    assert correct_a.shape == correct_b.shape and correct_a.ndim == 1, (
        f"correct mask shape 불일치: {correct_a.shape} vs {correct_b.shape}"
    )
    both_correct = int((correct_a & correct_b).sum())
    only_a = int((correct_a & ~correct_b).sum())     # A 만 정답 (= b)
    only_b = int((~correct_a & correct_b).sum())     # B 만 정답 (= c)
    both_wrong = int((~correct_a & ~correct_b).sum())

    table = [[both_correct, only_a],
             [only_b, both_wrong]]

    # disagreement 없으면 검정 불가능
    disagree = only_a + only_b
    if disagree == 0:
        return {
            "table": table,
            "both_correct": both_correct,
            "only_a": only_a,
            "only_b": only_b,
            "both_wrong": both_wrong,
            "chi2": 0.0,
            "p_value": 1.0,
            "method": "no-disagreement",
            "significant_0_05": False,
        }

    # 1순위: statsmodels
    try:
        from statsmodels.stats.contingency_tables import mcnemar  # type: ignore
        # exact=False → chi2 근사, correction=True → continuity correction
        res = mcnemar(np.asarray(table), exact=False, correction=True)
        chi2 = float(res.statistic)
        p_value = float(res.pvalue)
        method = "statsmodels.mcnemar(exact=False, correction=True)"
    except Exception:
        # 2순위: scipy 로 직접 계산
        try:
            from scipy.stats import chi2 as _chi2  # type: ignore
            b, c = only_a, only_b
            chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
            p_value = float(1.0 - _chi2.cdf(chi2, df=1))
            method = "scipy chi2 manual (continuity corrected)"
        except Exception as e:
            raise ImportError(
                "McNemar 계산용 statsmodels 또는 scipy 필요: "
                "pip install statsmodels scipy"
            ) from e

    return {
        "table": table,
        "both_correct": both_correct,
        "only_a": only_a,
        "only_b": only_b,
        "both_wrong": both_wrong,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "method": method,
        "significant_0_05": bool(p_value < 0.05),
    }


# ------------------------------------------------------------------
# 메트릭 (overall + per-class)
# ------------------------------------------------------------------

def overall_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """val_acc / macro_f1 / nll.  probs: (N, 4), y: (N,) int."""
    assert probs.shape[0] == y.shape[0]
    assert probs.shape[1] == NUM_CLASSES
    from sklearn.metrics import f1_score

    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, labels=list(range(NUM_CLASSES)),
                        average="macro", zero_division=0))
    idx = np.arange(len(y))
    p_true = probs[idx, y]
    nll = float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())
    return {"acc": acc, "macro_f1": f1, "nll": nll}


def per_class_report(probs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """per-class precision / recall / F1  — 각 shape (4,). sklearn zero_division=0."""
    from sklearn.metrics import precision_recall_fscore_support

    pred = probs.argmax(axis=1)
    # labels 고정 → 항상 (4,) 반환 (support 0 클래스도 0 으로 채움)
    p, r, f, _ = precision_recall_fscore_support(
        y, pred,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    return np.asarray(p), np.asarray(r), np.asarray(f)


def confmat(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """confusion matrix (행=true, 열=pred), labels=0..3 고정."""
    from sklearn.metrics import confusion_matrix
    pred = probs.argmax(axis=1)
    return confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))


# ------------------------------------------------------------------
# confusion matrix PNG (A, B 나란히)
# ------------------------------------------------------------------

def save_confmat_png(
    cm_a: np.ndarray,
    cm_b: np.ndarray,
    name_a: str,
    name_b: str,
    output_path: Path,
) -> None:
    """두 confusion matrix 를 좌/우 subplot 으로 저장. 각 셀에 수 기재.

    matplotlib.pyplot 만 사용 (seaborn X). 색: Blues.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in [(axes[0], cm_a, name_a), (axes[1], cm_b, name_b)]:
        cm = np.asarray(cm, dtype=np.int64)
        # normalize by row (true) for color — 절대값은 annotation 에만
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sum > 0, cm / np.clip(row_sum, 1, None), 0.0)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{title}\nacc={cm.trace() / max(1, cm.sum()):.4f}")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        # 셀에 숫자 기입
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = int(cm[i, j])
                frac = float(cm_norm[i, j])
                # 배경이 진하면 흰 글씨
                color = "white" if frac > 0.5 else "black"
                ax.text(j, i, f"{val}\n({frac:.2f})",
                        ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"[save png] {output_path}")


# ------------------------------------------------------------------
# markdown 리포트
# ------------------------------------------------------------------

def build_markdown(
    name_a: str,
    name_b: str,
    model_a_path: Path,
    model_b_path: Path,
    val_count: int,
    crop_mode: str,
    overall_a: dict,
    overall_b: dict,
    per_class_a: Tuple[np.ndarray, np.ndarray, np.ndarray],
    per_class_b: Tuple[np.ndarray, np.ndarray, np.ndarray],
    cm_a: np.ndarray,
    cm_b: np.ndarray,
    mc: dict,
    png_path: Path,
    seed: int,
) -> str:
    """비교 리포트 markdown 문자열."""
    lines: List[str] = []

    lines.append(f"# Model Comparison: {name_a} vs {name_b}")
    lines.append("")
    lines.append(f"- A: `{model_a_path.name}`  (path: `{model_a_path}`)")
    lines.append(f"- B: `{model_b_path.name}`  (path: `{model_b_path}`)")
    lines.append(f"- val samples: **{val_count}**  (crop_mode: `{crop_mode}`, seed: {seed})")
    lines.append(f"- confusion matrix png: `{png_path.name}`")
    lines.append("")

    # --- Overall ---
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | " + name_a + " | " + name_b + " | Δ (B − A) |")
    lines.append("|---|---:|---:|---:|")

    def _row(label: str, va: float, vb: float, higher_better: bool = True) -> str:
        d = vb - va
        sign = "+" if d >= 0 else ""
        return f"| {label} | {va:.4f} | {vb:.4f} | {sign}{d:.4f} |"

    lines.append(_row("val_acc", overall_a["acc"], overall_b["acc"]))
    lines.append(_row("macro_F1", overall_a["macro_f1"], overall_b["macro_f1"]))
    lines.append(_row("NLL (↓ better)", overall_a["nll"], overall_b["nll"], higher_better=False))
    lines.append("")

    # --- Sample-level delta (McNemar 2x2 입력도 이 값) ---
    lines.append("## Sample-level delta")
    lines.append("")
    lines.append(f"- 둘 다 정답: **{mc['both_correct']}**")
    lines.append(f"- {name_a} 만 맞춤 (A only): **{mc['only_a']}**")
    lines.append(f"- {name_b} 만 맞춤 (B only): **{mc['only_b']}**")
    lines.append(f"- 둘 다 오답: **{mc['both_wrong']}**")
    net = mc["only_b"] - mc["only_a"]
    sign = "+" if net >= 0 else ""
    lines.append(f"- **Net 개선 (B − A)**: {sign}{net} samples")
    lines.append("")

    # --- McNemar ---
    lines.append("## McNemar test")
    lines.append("")
    lines.append(f"- method: {mc['method']}")
    lines.append(f"- 2×2 contingency table (rows: A correct y/n, cols: B correct y/n):")
    lines.append("")
    lines.append("  |  | B 정답 | B 오답 |")
    lines.append("  |---|---:|---:|")
    lines.append(f"  | **A 정답** | {mc['both_correct']} | {mc['only_a']} |")
    lines.append(f"  | **A 오답** | {mc['only_b']} | {mc['both_wrong']} |")
    lines.append("")
    lines.append(f"- χ² = **{mc['chi2']:.4f}**,  p-value = **{mc['p_value']:.4g}**")
    if mc["method"] == "no-disagreement":
        lines.append("- 두 모델이 모든 샘플에서 동일한 정오판정 → 검정 불가 (p=1.0)")
    elif mc["significant_0_05"]:
        lines.append("- **p < 0.05 → 두 모델 성능 차이 통계적으로 유의**")
    else:
        lines.append("- p ≥ 0.05 → 통계적 유의성 없음 (차이 우연 범위)")
    lines.append("")

    # --- Per-class ---
    lines.append("## Per-class precision / recall / F1")
    lines.append("")
    pa, ra, fa = per_class_a
    pb, rb, fb = per_class_b
    lines.append(f"| class | P_{name_a} | R_{name_a} | F1_{name_a} | "
                 f"P_{name_b} | R_{name_b} | F1_{name_b} | ΔF1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, c in enumerate(CLASSES):
        df1 = float(fb[i] - fa[i])
        sign = "+" if df1 >= 0 else ""
        lines.append(
            f"| {c} "
            f"| {pa[i]:.4f} | {ra[i]:.4f} | {fa[i]:.4f} "
            f"| {pb[i]:.4f} | {rb[i]:.4f} | {fb[i]:.4f} "
            f"| {sign}{df1:.4f} |"
        )
    lines.append("")

    # --- Confusion matrices (raw) ---
    lines.append("## Confusion matrices (raw counts; rows=true, cols=pred)")
    lines.append("")
    for name, cm in [(name_a, cm_a), (name_b, cm_b)]:
        lines.append(f"### {name}")
        lines.append("")
        header = "| true ＼ pred | " + " | ".join(CLASSES) + " | total |"
        lines.append(header)
        lines.append("|---|" + "---:|" * (NUM_CLASSES + 1))
        cm = np.asarray(cm, dtype=np.int64)
        for i, c in enumerate(CLASSES):
            row_total = int(cm[i].sum())
            cells = " | ".join(str(int(cm[i, j])) for j in range(NUM_CLASSES))
            lines.append(f"| **{c}** | {cells} | {row_total} |")
        lines.append("")

    # --- Interpretation hint ---
    lines.append("## Notes")
    lines.append("")
    lines.append("- NLL 은 낮을수록 좋음 (calibration + 정답 확률). val_acc 와 별개로 확인 권장.")
    lines.append("- McNemar test 는 paired 모델 비교용 표준 검정. "
                 "b = A only, c = B only → (|b−c|−1)² / (b+c) ≈ χ²(df=1).")
    lines.append("- ΔF1 가 크게 개선된 클래스가 튜닝의 주요 이득 지점. "
                 "역으로 음수인 클래스는 regression 의심 (재확인 필요).")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="두 모델 val set 비교: overall/per-class/confmat/McNemar"
    )
    ap.add_argument("--models-a", required=True,
                    help="모델 A 경로 (.h5/.pt). 튜닝 전 baseline 권장.")
    ap.add_argument("--models-b", required=True,
                    help="모델 B 경로 (.h5/.pt). 튜닝 후 비교 대상.")
    ap.add_argument("--name-a", default=None,
                    help="보고서에 표기할 A 이름 (default: 파일 stem)")
    ap.add_argument("--name-b", default=None,
                    help="보고서에 표기할 B 이름 (default: 파일 stem)")
    ap.add_argument("--val-dir", required=True,
                    help="val 이미지 디렉토리 (하위 anger/happy/panic/sadness)")
    ap.add_argument("--val-label-root", default=None,
                    help="annot_A bbox 로드용 라벨 루트 (예: data_rot/label). "
                         "미지정 시 --val-dir 의 ../label 자동 감지.")
    ap.add_argument("--crop-mode", default="bbox", choices=["bbox", "raw"],
                    help="bbox: annot_A bbox 로 crop 후 평가 (default, 학습조건 일치). "
                         "raw: 전체 이미지.")
    ap.add_argument("--output-dir", default=str(PROJECT / "results"),
                    help="리포트/PNG 저장 폴더")
    ap.add_argument("--cache-dir", default=str(PROJECT / "results" / "ensemble_cache"),
                    help="per-model probs npz 캐시 폴더 (ensemble_search 와 공유)")
    ap.add_argument("--force-cpu", action="store_true",
                    help="TF/torch CPU 강제")
    ap.add_argument("--force-recompute", action="store_true",
                    help="캐시 무시하고 전부 재예측")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="클래스당 최대 이미지 수 (0=제한없음). 디버그용.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # -------- force_cpu 실제 적용 --------
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _log("force_cpu: CUDA_VISIBLE_DEVICES=''")

    # -------- 경로 정규화 --------
    def _resolve_model_path(raw: str) -> Path:
        p = Path(raw)
        if not p.is_absolute():
            cand = PROJECT / p
            p = cand if cand.is_file() else p
        if not p.is_file():
            raise FileNotFoundError(f"모델 파일 없음: {p}")
        return p.resolve()

    model_a = _resolve_model_path(args.models_a)
    model_b = _resolve_model_path(args.models_b)
    if model_a == model_b:
        raise ValueError(f"모델 A 와 B 가 동일: {model_a}")
    name_a = args.name_a or model_a.stem
    name_b = args.name_b or model_b.stem

    val_dir = Path(args.val_dir)
    if not val_dir.is_absolute():
        val_dir = PROJECT / val_dir
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT / cache_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------- label_root 자동 감지 (bbox 모드) --------
    label_root: Optional[Path] = None
    crop_mode = args.crop_mode
    if crop_mode == "bbox":
        if args.val_label_root:
            label_root = Path(args.val_label_root)
            if not label_root.is_absolute():
                label_root = PROJECT / label_root
        else:
            candidate = val_dir.parent.parent / "label"
            if candidate.is_dir():
                label_root = candidate
                _log(f"[auto] label_root = {label_root}")
            else:
                _log(f"[warn] label_root 자동감지 실패 ({candidate}) → crop_mode 'raw' 로 강등")
                crop_mode = "raw"

    # -------- val 수집 --------
    paths, y_true, bboxes = es.collect_val_samples(
        val_dir,
        limit_per_class=args.limit_per_class,
        label_root=label_root,
    )
    if len(paths) == 0:
        raise RuntimeError("val 이미지 0장 — 디렉토리/구조 확인")

    _log(f"[crop_mode] {crop_mode}")
    _log(f"[model A] {model_a}")
    _log(f"[model B] {model_b}")

    # -------- probs 수집 --------
    t0 = time.time()
    probs_a = get_probs(
        model_a, paths, bboxes, cache_dir,
        crop_mode=crop_mode,
        force_cpu=args.force_cpu,
        force_recompute=args.force_recompute,
    )
    _log(f"[probs A] shape={probs_a.shape}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    probs_b = get_probs(
        model_b, paths, bboxes, cache_dir,
        crop_mode=crop_mode,
        force_cpu=args.force_cpu,
        force_recompute=args.force_recompute,
    )
    _log(f"[probs B] shape={probs_b.shape}  ({time.time()-t0:.1f}s)")

    # -------- 메트릭 --------
    overall_a = overall_metrics(probs_a, y_true)
    overall_b = overall_metrics(probs_b, y_true)
    _log(f"[overall A] {overall_a}")
    _log(f"[overall B] {overall_b}")

    pcr_a = per_class_report(probs_a, y_true)
    pcr_b = per_class_report(probs_b, y_true)

    cm_a = confmat(probs_a, y_true)
    cm_b = confmat(probs_b, y_true)

    # sanity: confmat trace / acc 일치 확인 (bug 0 증빙)
    acc_from_cm_a = float(cm_a.trace() / max(1, cm_a.sum()))
    acc_from_cm_b = float(cm_b.trace() / max(1, cm_b.sum()))
    if abs(acc_from_cm_a - overall_a["acc"]) > 1e-6:
        raise RuntimeError(
            f"cm trace acc A {acc_from_cm_a} ≠ argmax acc {overall_a['acc']}"
        )
    if abs(acc_from_cm_b - overall_b["acc"]) > 1e-6:
        raise RuntimeError(
            f"cm trace acc B {acc_from_cm_b} ≠ argmax acc {overall_b['acc']}"
        )

    # -------- McNemar --------
    correct_a = (probs_a.argmax(axis=1) == y_true)
    correct_b = (probs_b.argmax(axis=1) == y_true)
    mc = mcnemar_test(correct_a, correct_b)
    _log(f"[mcnemar] chi2={mc['chi2']:.4f}  p={mc['p_value']:.4g}  "
         f"(onlyA={mc['only_a']}, onlyB={mc['only_b']}, method={mc['method']})")

    # -------- 출력 파일명 --------
    stem = f"compare_{name_a}_vs_{name_b}"
    md_path = output_dir / f"{stem}.md"
    png_path = output_dir / f"{stem}_confmat.png"

    # -------- confmat PNG --------
    save_confmat_png(cm_a, cm_b, name_a, name_b, png_path)

    # -------- markdown --------
    md = build_markdown(
        name_a=name_a, name_b=name_b,
        model_a_path=model_a, model_b_path=model_b,
        val_count=len(paths),
        crop_mode=crop_mode,
        overall_a=overall_a, overall_b=overall_b,
        per_class_a=pcr_a, per_class_b=pcr_b,
        cm_a=cm_a, cm_b=cm_b,
        mc=mc,
        png_path=png_path,
        seed=args.seed,
    )
    md_path.write_text(md, encoding="utf-8")
    _log(f"[save md] {md_path}")

    # -------- JSON dump (부가) — 프로그래매틱 접근용 --------
    json_path = output_dir / f"{stem}.json"
    json_payload = {
        "name_a": name_a,
        "name_b": name_b,
        "model_a": str(model_a),
        "model_b": str(model_b),
        "val_count": len(paths),
        "crop_mode": crop_mode,
        "overall_a": overall_a,
        "overall_b": overall_b,
        "per_class": {
            "precision_a": [float(x) for x in pcr_a[0]],
            "recall_a":    [float(x) for x in pcr_a[1]],
            "f1_a":        [float(x) for x in pcr_a[2]],
            "precision_b": [float(x) for x in pcr_b[0]],
            "recall_b":    [float(x) for x in pcr_b[1]],
            "f1_b":        [float(x) for x in pcr_b[2]],
            "classes":     CLASSES,
        },
        "confmat_a": cm_a.tolist(),
        "confmat_b": cm_b.tolist(),
        "mcnemar": {
            "chi2":         mc["chi2"],
            "p_value":      mc["p_value"],
            "method":       mc["method"],
            "both_correct": mc["both_correct"],
            "only_a":       mc["only_a"],
            "only_b":       mc["only_b"],
            "both_wrong":   mc["both_wrong"],
            "table":        mc["table"],
            "significant_0_05": mc["significant_0_05"],
        },
        "seed": args.seed,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    _log(f"[save json] {json_path}")

    # -------- 최종 summary 한 줄 --------
    da = overall_b["acc"] - overall_a["acc"]
    df1 = overall_b["macro_f1"] - overall_a["macro_f1"]
    dn = overall_b["nll"] - overall_a["nll"]
    sig = "YES" if mc["significant_0_05"] else "no"
    _log(f"[SUMMARY] {name_a} → {name_b}  "
         f"Δacc={da:+.4f}  Δmacro_f1={df1:+.4f}  Δnll={dn:+.4f}  "
         f"McNemar p={mc['p_value']:.4g} (sig<0.05: {sig})")
    _log("[done]")


if __name__ == "__main__":
    main()
