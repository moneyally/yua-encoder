"""3인 라벨러 일치성 EDA — 전처리/EDA 배점 공략 (EDA 15 + 전처리 15).

설계 요약
──────────────────────────────────────────────────────────────
데이터:
  /workspace/user4/emotion-project/data_rot/label/{train,val}/{split}_{cls}.json (euc-kr)
  각 항목 = {filename, faceExp_uploader, annot_A, annot_B, annot_C, ...}
  annot_* = {boxes:{minX,maxX,minY,maxY}, faceExp:"분노|기쁨|당황|슬픔", bg}

분석 항목:
  1) faceExp 일치율 (전원일치/다수결/완전불일치) — 전체·클래스별
  2) pairwise faceExp confusion (AB, AC, BC)
  3) 폴더 클래스(GT) 대비 라벨러 답 매칭률
  4) bbox IoU: AB / AC / BC 평균·분포·클래스별 박스플롯
  5) 평균 IoU 낮은 top10 (가장 논쟁적인 bbox)
  6) 3인 faceExp 모두 다른 샘플 10장 이미지 그리드 (A=red, B=green, C=blue)

산출물:
  results/annot_consistency.md       — 수치 표 + 인사이트
  results/annot_disagreement_grid.png — 논쟁 샘플 그리드
  results/annot_iou_boxplot.png       — IoU 박스플롯

구현 주의:
  - json: euc-kr 우선, utf-8 fallback
  - koreanize_matplotlib 없으면 FACEEXP_FALLBACK 영문 라벨
  - annot 중 하나라도 없거나 bbox dict 누락이면 skip (로그 집계만)
  - 음수 bbox 좌표 → [0,W]×[0,H] clip
  - data_rot은 정규화되어 있으므로 ImageOps.exif_transpose 불필요
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import koreanize_matplotlib  # noqa: F401
    _HANGUL_OK = True
except Exception as _e:
    _HANGUL_OK = False
    print(f"[warn] koreanize_matplotlib 미설치 → 한글은 FACEEXP_FALLBACK 매핑 ({_e})")

# ─────────────────────────────────────────────────────────────
CLASSES = ["anger", "happy", "panic", "sadness"]
CLASS_KOR = {"anger": "분노", "happy": "기쁨", "panic": "당황", "sadness": "슬픔"}
# folder class ↔ faceExp 값 매핑 (GT 정답지)
CLASS_TO_FACEEXP = {"anger": "분노", "happy": "기쁨", "panic": "당황", "sadness": "슬픔"}
SPLITS = ["train", "val"]

FACEEXP_FALLBACK = {
    "분노": "anger", "화남": "anger",
    "기쁨": "happy", "행복": "happy",
    "당황": "panic", "놀람": "surprise", "공포": "fear",
    "슬픔": "sadness", "중립": "neutral",
    "상처": "hurt", "불안": "anxiety", "혐오": "disgust",
}


def _disp(label: str) -> str:
    """한글 폰트 없을 때 fallback 영문 라벨."""
    if _HANGUL_OK:
        return label
    return FACEEXP_FALLBACK.get(label, label)


# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3인 라벨러 일치성 EDA")
    default_root = os.environ.get("PROJECT_ROOT", "/workspace/user4/emotion-project")
    ap.add_argument("--root", default=default_root, help="프로젝트 루트")
    ap.add_argument("--data-subdir", default="data_rot", help="data 또는 data_rot")
    ap.add_argument("--results-subdir", default="results", help="결과 디렉토리")
    ap.add_argument("--split", default="train", choices=["train", "val", "both"],
                    help="분석 대상 split")
    ap.add_argument("--topk", type=int, default=10, help="논쟁 이미지 top K")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────
def load_labels(lbl_root: Path, splits: List[str]) -> Dict[str, Dict[str, list]]:
    per: Dict[str, Dict[str, list]] = {s: {c: [] for c in CLASSES} for s in splits}
    for s in splits:
        for c in CLASSES:
            p = lbl_root / s / f"{s}_{c}.json"
            if not p.is_file():
                print(f"[warn] {p} 없음")
                continue
            loaded = None
            last_err: Optional[Exception] = None
            for enc in ("euc-kr", "utf-8"):
                try:
                    with open(p, encoding=enc) as f:
                        loaded = json.load(f)
                    break
                except (UnicodeDecodeError, json.JSONDecodeError, OSError) as e:
                    last_err = e
            if loaded is None:
                print(f"[warn] {p.name} 로드 실패: {last_err}")
                continue
            per[s][c] = loaded
    return per


# ─────────────────────────────────────────────────────────────
# bbox & IoU
# ─────────────────────────────────────────────────────────────
def get_bbox(annot: dict, W: Optional[int] = None, H: Optional[int] = None
             ) -> Optional[Tuple[float, float, float, float]]:
    """annot → (x0,y0,x1,y1). 좌표 clip, 유효 검사."""
    if not isinstance(annot, dict):
        return None
    b = annot.get("boxes")
    if not isinstance(b, dict):
        return None
    try:
        x0 = float(b["minX"]); x1 = float(b["maxX"])
        y0 = float(b["minY"]); y1 = float(b["maxY"])
    except (KeyError, TypeError, ValueError):
        return None
    # clip 음수 → [0, W/H]
    if W is not None:
        x0 = max(0.0, min(x0, W)); x1 = max(0.0, min(x1, W))
    else:
        x0 = max(0.0, x0); x1 = max(0.0, x1)
    if H is not None:
        y0 = max(0.0, min(y0, H)); y1 = max(0.0, min(y1, H))
    else:
        y0 = max(0.0, y0); y1 = max(0.0, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


def face_exp(annot: dict) -> Optional[str]:
    if not isinstance(annot, dict):
        return None
    v = annot.get("faceExp")
    return v if isinstance(v, str) and v else None


# ─────────────────────────────────────────────────────────────
# 분석
# ─────────────────────────────────────────────────────────────
def analyze(per_lbl: Dict[str, Dict[str, list]], splits: List[str]
            ) -> dict:
    """주요 통계 누적."""
    stats = {
        "n_items": 0,
        "n_valid_faceExp": 0,  # annot_A/B/C 모두 faceExp 유효
        "n_valid_bbox": 0,     # annot_A/B/C 모두 bbox 유효
        "n_skip_faceExp": 0,
        "n_skip_bbox": 0,
        "per_class_valid": Counter(),
        "per_class_all_agree": Counter(),     # 3인 완전일치
        "per_class_majority": Counter(),      # 2/3 이상 일치 (전원일치 포함)
        "per_class_all_diff": Counter(),      # 3인 모두 다름
        "per_class_gt_match_any": Counter(),  # 폴더=GT, 라벨러 중 1명이라도 일치
        "per_class_gt_match_all": Counter(),  # 3인 모두 GT와 일치
        "pair_conf": Counter(),  # (labelA, labelB) frozenset 쌍 confusion
        "iou_ab": [], "iou_ac": [], "iou_bc": [],
        "iou_per_class": {c: [] for c in CLASSES},  # mean IoU 3쌍 평균
        "low_iou_samples": [],  # (mean_iou, split, cls, filename, boxes_dict, exps_tuple)
        "disagree_samples": [],  # 3인 다 다름: (split, cls, filename, boxes, exps)
    }

    for s in splits:
        for c in CLASSES:
            gt = CLASS_TO_FACEEXP[c]
            items = per_lbl[s].get(c, [])
            for it in items:
                if not isinstance(it, dict):
                    continue
                stats["n_items"] += 1
                fname = it.get("filename", "")
                a, b, d = it.get("annot_A"), it.get("annot_B"), it.get("annot_C")

                # ── faceExp
                ea, eb, ec = face_exp(a), face_exp(b), face_exp(d)
                if ea and eb and ec:
                    stats["n_valid_faceExp"] += 1
                    stats["per_class_valid"][c] += 1
                    exps = (ea, eb, ec)
                    uniq = set(exps)
                    if len(uniq) == 1:
                        stats["per_class_all_agree"][c] += 1
                        stats["per_class_majority"][c] += 1
                    elif len(uniq) == 2:
                        stats["per_class_majority"][c] += 1  # 2/3 다수결 존재
                    else:  # 3 모두 다름
                        stats["per_class_all_diff"][c] += 1

                    # GT 매칭
                    matches_gt = sum(1 for e in exps if e == gt)
                    if matches_gt >= 1:
                        stats["per_class_gt_match_any"][c] += 1
                    if matches_gt == 3:
                        stats["per_class_gt_match_all"][c] += 1

                    # pairwise confusion (순서 무관)
                    for x, y in ((ea, eb), (ea, ec), (eb, ec)):
                        if x != y:
                            stats["pair_conf"][frozenset((x, y))] += 1

                    if len(uniq) == 3:
                        # 논쟁 샘플 후보
                        ba = get_bbox(a); bb = get_bbox(b); bc = get_bbox(d)
                        stats["disagree_samples"].append({
                            "split": s, "cls": c, "filename": fname,
                            "boxes": (ba, bb, bc), "exps": exps,
                        })
                else:
                    stats["n_skip_faceExp"] += 1

                # ── bbox IoU
                ba, bb, bc = get_bbox(a), get_bbox(b), get_bbox(d)
                if ba and bb and bc:
                    stats["n_valid_bbox"] += 1
                    i_ab = iou(ba, bb); i_ac = iou(ba, bc); i_bc = iou(bb, bc)
                    stats["iou_ab"].append(i_ab)
                    stats["iou_ac"].append(i_ac)
                    stats["iou_bc"].append(i_bc)
                    mean_i = (i_ab + i_ac + i_bc) / 3
                    stats["iou_per_class"][c].append(mean_i)
                    stats["low_iou_samples"].append(
                        (mean_i, s, c, fname, (ba, bb, bc),
                         (ea or "-", eb or "-", ec or "-"))
                    )
                else:
                    stats["n_skip_bbox"] += 1

    stats["low_iou_samples"].sort(key=lambda x: x[0])
    return stats


# ─────────────────────────────────────────────────────────────
# 플롯
# ─────────────────────────────────────────────────────────────
def plot_iou_boxplot(stats: dict, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) 3 쌍 전체 분포
    ax = axes[0]
    data = [stats["iou_ab"], stats["iou_ac"], stats["iou_bc"]]
    labels = ["A vs B", "A vs C", "B vs C"]
    if any(len(x) for x in data):
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        except TypeError:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        for p, col in zip(bp["boxes"], colors):
            p.set_facecolor(col); p.set_alpha(0.7)
        ax.set_ylabel("IoU")
        ax.set_ylim(0, 1.02)
        ax.set_title("Pairwise bbox IoU between 3 annotators")
        ax.grid(axis="y", alpha=0.3)
        # 평균 텍스트
        for i, arr in enumerate(data, 1):
            if arr:
                ax.text(i, 0.02, f"mean={np.mean(arr):.3f}\nn={len(arr)}",
                        ha="center", fontsize=8,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax.set_title("no IoU data"); ax.axis("off")

    # (b) 클래스별 mean IoU
    ax = axes[1]
    data2 = [stats["iou_per_class"][c] for c in CLASSES]
    if any(len(x) for x in data2):
        try:
            bp = ax.boxplot(data2, tick_labels=CLASSES, patch_artist=True, showmeans=True)
        except TypeError:
            bp = ax.boxplot(data2, labels=CLASSES, patch_artist=True, showmeans=True)
        colors = ["#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
        for p, col in zip(bp["boxes"], colors):
            p.set_facecolor(col); p.set_alpha(0.7)
        ax.set_ylabel("mean IoU (A/B/C 3-pair avg)")
        ax.set_ylim(0, 1.02)
        ax.set_title("Per-class mean pairwise IoU")
        ax.grid(axis="y", alpha=0.3)
        for i, arr in enumerate(data2, 1):
            if arr:
                ax.text(i, 0.02, f"μ={np.mean(arr):.3f}\nn={len(arr)}",
                        ha="center", fontsize=8,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax.set_title("no per-class IoU"); ax.axis("off")

    fig.suptitle("Annotator bbox consistency (IoU)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_disagreement_grid(stats: dict, img_root: Path, outpath: Path, topk: int = 10) -> int:
    """3인 faceExp 모두 다른 샘플 topk장 이미지 그리드.
    annot_A=red, B=green, C=blue 로 bbox 오버레이.
    """
    samples = stats["disagree_samples"][:topk]
    # 부족하면 pairwise 불일치 + 낮은 IoU 로 보충
    if len(samples) < topk:
        print(f"[info] 3인 faceExp 완전 불일치 샘플 부족 ({len(samples)}). "
              f"낮은 IoU 샘플로 보충.")
        seen = {(s["split"], s["cls"], s["filename"]) for s in samples}
        for t in stats["low_iou_samples"]:
            k = (t[1], t[2], t[3])
            if k in seen:
                continue
            samples.append({
                "split": t[1], "cls": t[2], "filename": t[3],
                "boxes": t[4], "exps": t[5],
            })
            seen.add(k)
            if len(samples) >= topk:
                break

    if not samples:
        print("[warn] 시각화할 샘플 없음 — 빈 플레이스홀더 PNG")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no disagreement samples", ha="center", va="center")
        ax.axis("off")
        fig.savefig(outpath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return 0

    n = len(samples)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 4.0))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    annot_colors = ["red", "lime", "cyan"]
    annot_names = ["A", "B", "C"]

    for idx, s in enumerate(samples):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        img_path = img_root / s["split"] / s["cls"] / s["filename"]
        if not img_path.is_file():
            ax.text(0.5, 0.5, f"missing:\n{s['filename'][:20]}",
                    ha="center", va="center", fontsize=8)
            ax.axis("off"); continue
        try:
            with Image.open(img_path) as im:
                im.load()
                im = im.convert("RGB")
                ax.imshow(np.array(im))
                W, H = im.size
        except (UnidentifiedImageError, OSError) as e:
            ax.text(0.5, 0.5, f"err: {type(e).__name__}",
                    ha="center", va="center", fontsize=8)
            ax.axis("off"); continue

        for bi, (box, name, col, exp) in enumerate(zip(
                s["boxes"], annot_names, annot_colors, s["exps"])):
            if box is None:
                continue
            x0, y0, x1, y1 = box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, ec=col, lw=2.0))
            # faceExp 라벨을 bbox 위에 (각 라벨러마다 y 옵셋)
            ax.text(x0 + 2, y0 - 6 - bi * 16,
                    f"{name}:{_disp(exp)}", color=col, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="black", ec=col, alpha=0.6))

        title = f"{s['cls']} / {s['filename'][:14]}…"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # 남는 축 끄기
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    # 범례
    legend_handles = [
        mpatches.Patch(color=annot_colors[0], label="annot_A"),
        mpatches.Patch(color=annot_colors[1], label="annot_B"),
        mpatches.Patch(color=annot_colors[2], label="annot_C"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    subtitle = ("(3인 faceExp 불일치 + 낮은 IoU 보충)" if _HANGUL_OK
                else "(3-annotator faceExp disagreement + low-IoU fill)")
    fig.suptitle(f"Top-{len(samples)} disagreement samples {subtitle}", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return len(samples)


# ─────────────────────────────────────────────────────────────
# 마크다운 리포트
# ─────────────────────────────────────────────────────────────
def write_markdown(stats: dict, topk_samples: list, outpath: Path,
                   splits: List[str], topk: int) -> None:
    L: List[str] = []
    L += [f"# 3인 라벨러 일치성 EDA (split={','.join(splits)})", ""]

    n_items = stats["n_items"]
    n_fe = stats["n_valid_faceExp"]
    n_bb = stats["n_valid_bbox"]
    L += [
        "## 데이터 요약",
        f"- 항목 수: **{n_items}**",
        f"- 3인 faceExp 유효: **{n_fe}** (skip {stats['n_skip_faceExp']})",
        f"- 3인 bbox 유효: **{n_bb}** (skip {stats['n_skip_bbox']})",
        "",
    ]

    # ── faceExp 일치율
    tot_valid = n_fe
    tot_all_agree = sum(stats["per_class_all_agree"].values())
    tot_majority = sum(stats["per_class_majority"].values())
    tot_all_diff = sum(stats["per_class_all_diff"].values())

    def pct(x, y):
        return 100.0 * x / y if y else 0.0

    L += [
        "## 1) faceExp 일치율",
        "",
        f"- 전원 일치 (3/3): **{tot_all_agree}/{tot_valid} = {pct(tot_all_agree, tot_valid):.2f}%**",
        f"- 다수결 일치 (≥2/3): **{tot_majority}/{tot_valid} = {pct(tot_majority, tot_valid):.2f}%**",
        f"- 3인 완전 불일치: **{tot_all_diff}/{tot_valid} = {pct(tot_all_diff, tot_valid):.2f}%**",
        "",
        "### 클래스별",
        "",
        "| 클래스 | 유효 n | 전원일치 % | 다수결 % | 완전불일치 % |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        v = stats["per_class_valid"][c]
        a = stats["per_class_all_agree"][c]
        m = stats["per_class_majority"][c]
        d = stats["per_class_all_diff"][c]
        L.append(f"| {c} ({CLASS_KOR[c]}) | {v} | {pct(a, v):.2f} | {pct(m, v):.2f} | {pct(d, v):.2f} |")
    L.append("")

    # ── pairwise confusion top
    pair = stats["pair_conf"].most_common(10)
    L += ["### 자주 헷갈리는 감정 쌍 (상위 10)", ""]
    if pair:
        L += ["| 쌍 | 횟수 |", "|---|---:|"]
        for fset, cnt in pair:
            a, b = list(fset) if len(fset) > 1 else (list(fset)[0], list(fset)[0])
            L.append(f"| {a} ↔ {b} | {cnt} |")
    else:
        L.append("(데이터 없음)")
    L.append("")

    # ── GT 매칭
    L += ["## 2) 폴더 클래스(GT) vs 라벨러 faceExp 매칭", "",
          "| 클래스 | 유효 n | 1명↑ 일치 % | 3명 모두 일치 % |",
          "|---|---:|---:|---:|"]
    for c in CLASSES:
        v = stats["per_class_valid"][c]
        any_ = stats["per_class_gt_match_any"][c]
        all_ = stats["per_class_gt_match_all"][c]
        L.append(f"| {c} ({CLASS_KOR[c]}) | {v} | {pct(any_, v):.2f} | {pct(all_, v):.2f} |")
    L.append("")

    # ── bbox IoU
    ab = stats["iou_ab"]; ac = stats["iou_ac"]; bc = stats["iou_bc"]

    def _stat(a):
        if not a:
            return ("-", "-", "-", "-", "-")
        arr = np.array(a)
        return (f"{arr.mean():.3f}", f"{np.median(arr):.3f}",
                f"{arr.min():.3f}", f"{arr.max():.3f}", f"{arr.std():.3f}")

    L += [
        "## 3) bbox IoU (annot pair별)",
        "",
        "| 쌍 | n | 평균 | 중앙 | 최소 | 최대 | std |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| A vs B | {len(ab)} | {_stat(ab)[0]} | {_stat(ab)[1]} | {_stat(ab)[2]} | {_stat(ab)[3]} | {_stat(ab)[4]} |",
        f"| A vs C | {len(ac)} | {_stat(ac)[0]} | {_stat(ac)[1]} | {_stat(ac)[2]} | {_stat(ac)[3]} | {_stat(ac)[4]} |",
        f"| B vs C | {len(bc)} | {_stat(bc)[0]} | {_stat(bc)[1]} | {_stat(bc)[2]} | {_stat(bc)[3]} | {_stat(bc)[4]} |",
        "",
        "### 클래스별 평균 IoU (3쌍 평균)",
        "",
        "| 클래스 | n | 평균 | 중앙 | 최소 |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        a = stats["iou_per_class"][c]
        if a:
            arr = np.array(a)
            L.append(f"| {c} | {len(a)} | {arr.mean():.3f} | {np.median(arr):.3f} | {arr.min():.3f} |")
        else:
            L.append(f"| {c} | 0 | - | - | - |")
    L.append("")

    # ── 논쟁 top-k
    L += [f"## 4) 평균 IoU 최저 top-{topk} (가장 논쟁적 bbox)", "",
          "| rank | mean_IoU | split/class | filename | (A,B,C) faceExp |",
          "|---:|---:|---|---|---|"]
    for i, (mean_i, s, c, fn, _boxes, exps) in enumerate(stats["low_iou_samples"][:topk], 1):
        L.append(f"| {i} | {mean_i:.3f} | {s}/{c} | `{fn}` | {exps[0]} / {exps[1]} / {exps[2]} |")
    L.append("")

    # 3인 완전 불일치 샘플 (시각화 대상)
    if topk_samples:
        L += [f"### 그리드 시각화에 포함된 샘플 ({len(topk_samples)}장)", "",
              "| # | split/class | filename | A / B / C |",
              "|---:|---|---|---|"]
        for i, sp in enumerate(topk_samples, 1):
            L.append(f"| {i} | {sp['split']}/{sp['cls']} | `{sp['filename']}` | "
                     f"{sp['exps'][0]} / {sp['exps'][1]} / {sp['exps'][2]} |")
        L.append("")

    # ── 인사이트
    mean_ab = np.mean(ab) if ab else 0
    mean_ac = np.mean(ac) if ac else 0
    mean_bc = np.mean(bc) if bc else 0
    L += [
        "## 5) 데이터 품질 인사이트",
        "",
        f"- 3인 bbox IoU 평균 {mean_ab:.3f}/{mean_ac:.3f}/{mean_bc:.3f}. "
        f"0.9 이상이면 라벨 일관성 우수 → annot_A bbox 단일 사용 정당화 가능. "
        f"0.7 미만이면 3명 평균 박스(median or union) 전처리 고려.",
        f"- 전원 일치율 {pct(tot_all_agree, tot_valid):.1f}%, 완전 불일치 {pct(tot_all_diff, tot_valid):.1f}%. "
        f"불일치가 큰 클래스는 모델 confusion 과도 직결 — loss weighting 또는 soft label 실험 후보.",
        f"- pairwise confusion 최다쌍 = " +
        (", ".join(f"{list(k)[0]}↔{list(k)[1] if len(k)>1 else list(k)[0]}({v})" for k, v in pair[:3])
         if pair else "(없음)") +
        ". 해당 쌍 간 구분이 모델에게도 어렵다는 신호 — feature aug(face crop, color jitter)로 보완.",
        "",
        "## 6) 산출물",
        "- `results/annot_consistency.md` (이 파일)",
        "- `results/annot_disagreement_grid.png` — 논쟁 샘플 그리드",
        "- `results/annot_iou_boxplot.png` — IoU 박스플롯",
    ]
    outpath.write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    root = Path(args.root)
    data = root / args.data_subdir
    lbl_root = data / "label"
    img_root = data / "img"
    out = root / args.results_subdir
    out.mkdir(exist_ok=True)

    if not lbl_root.is_dir():
        print(f"[!] {lbl_root} 없음"); sys.exit(2)

    splits = SPLITS if args.split == "both" else [args.split]

    print(f"[1/4] 라벨 로드 @ {lbl_root}  splits={splits}")
    per_lbl = load_labels(lbl_root, splits)
    for s in splits:
        for c in CLASSES:
            print(f"    {s}/{c}: {len(per_lbl[s][c])}")

    print("[2/4] 일치성 분석 ...")
    stats = analyze(per_lbl, splits)
    print(f"    valid_faceExp={stats['n_valid_faceExp']}, "
          f"valid_bbox={stats['n_valid_bbox']}, "
          f"skip_fe={stats['n_skip_faceExp']}, skip_bb={stats['n_skip_bbox']}")

    print("[3/4] 플롯 생성 ...")
    plot_iou_boxplot(stats, out / "annot_iou_boxplot.png")
    n_drawn = plot_disagreement_grid(
        stats, img_root, out / "annot_disagreement_grid.png", topk=args.topk)
    print(f"    disagreement grid: {n_drawn} samples")

    # 그리드에 실제 들어간 샘플 목록 (보고용)
    topk_samples = stats["disagree_samples"][:args.topk]

    print("[4/4] 마크다운 리포트 ...")
    write_markdown(stats, topk_samples, out / "annot_consistency.md",
                   splits, args.topk)

    # ── 콘솔 요약
    tot_v = stats["n_valid_faceExp"]
    tot_a = sum(stats["per_class_all_agree"].values())
    tot_m = sum(stats["per_class_majority"].values())
    tot_d = sum(stats["per_class_all_diff"].values())
    ab = stats["iou_ab"]; ac = stats["iou_ac"]; bc = stats["iou_bc"]

    print("\n=== annot_consistency 완료 ===")
    print(f"전원일치: {tot_a}/{tot_v} = {100*tot_a/tot_v:.2f}%" if tot_v else "전원일치: -")
    print(f"다수결  : {tot_m}/{tot_v} = {100*tot_m/tot_v:.2f}%" if tot_v else "다수결: -")
    print(f"완전불일치: {tot_d}/{tot_v} = {100*tot_d/tot_v:.2f}%" if tot_v else "완전불일치: -")
    print(f"mean IoU AB={np.mean(ab):.3f} AC={np.mean(ac):.3f} BC={np.mean(bc):.3f}"
          if ab else "IoU: -")
    print(f"low-IoU top {args.topk}:")
    for i, (mean_i, s, c, fn, _, exps) in enumerate(stats["low_iou_samples"][:args.topk], 1):
        print(f"  {i:2d}. IoU={mean_i:.3f} {s}/{c} {fn}  ({exps[0]}/{exps[1]}/{exps[2]})")
    print(f"\n산출물: {out}")
    for name in ("annot_consistency.md", "annot_disagreement_grid.png", "annot_iou_boxplot.png"):
        p = out / name
        if p.is_file():
            print(f"  - {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
