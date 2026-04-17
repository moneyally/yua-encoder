"""데이터 검증 + EDA 통합 스크립트.

산출물:
  results/eda_report.png     — 분포 차트 6개 (클래스·해상도·종횡비·연령·성별·faceExp 매트릭스)
  results/samples_grid.png   — 클래스별 샘플 이미지 + bbox 오버레이
  results/seg_overlay.png    — segmentation mask 시각화 샘플
  results/bbox_analysis.png  — bbox 크기/종횡비 박스플롯
  results/eda_summary.md     — 수치/요약 리포트

평가 항목(EDA 15점) 대응: 클래스 분포, 해상도, 샘플, seg, bbox, 나이/성별, 라벨 매칭, 깨진 이미지.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# 한글 폰트 자동 설정 (koreanize_matplotlib 번들 NanumGothic)
try:
    import koreanize_matplotlib  # noqa: F401
    _HANGUL_OK = True
except Exception as _e:
    _HANGUL_OK = False
    print(f"[warn] koreanize_matplotlib 로드 실패 → 한글 라벨은 로마자 매핑으로 대체 ({_e})")

# ──────────────────────────────────────────────────────────────
# 기본 상수 + CLI
# ──────────────────────────────────────────────────────────────
CLASSES = ["anger", "happy", "panic", "sadness"]
CLASS_KOR = {"anger": "화남", "happy": "행복", "panic": "놀람/공포", "sadness": "슬픔"}
SPLITS = ["train", "val"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# faceExp_uploader 한국어값을 폰트 깨짐 대비 영문 매핑 (fallback)
FACEEXP_FALLBACK = {
    "분노": "anger(분노)", "화남": "anger(화남)",
    "기쁨": "happy(기쁨)", "행복": "happy(행복)",
    "당황": "surprise(당황)", "놀람": "surprise(놀람)", "공포": "fear(공포)",
    "슬픔": "sadness(슬픔)", "중립": "neutral(중립)",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="데이터 검증 + EDA 통합")
    default_root = os.environ.get("PROJECT_ROOT", "/workspace/user4/emotion-project")
    ap.add_argument("--root", default=default_root, help="프로젝트 루트 (data/, results/ 포함 경로)")
    ap.add_argument("--data-subdir", default="data",
                    help="루트 아래 데이터 서브폴더 (예: data, data_rot)")
    ap.add_argument("--results-subdir", default="results",
                    help="루트 아래 결과 서브폴더")
    ap.add_argument("--tag", default="",
                    help="산출 파일명 prefix (예: 'rot_' → rot_eda_report.png)")
    ap.add_argument("--samples-per-class", type=int, default=4, help="samples_grid.png 클래스당 샘플 수")
    ap.add_argument("--size-sample", type=int, default=300, help="해상도 분포용 클래스별 샘플 수")
    return ap.parse_args()


# ──────────────────────────────────────────────────────────────
# 스캐너
# ──────────────────────────────────────────────────────────────
def scan_images(img_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    per: Dict[str, Dict[str, List[Path]]] = {s: {c: [] for c in CLASSES} for s in SPLITS}
    for s in SPLITS:
        for c in CLASSES:
            d = img_root / s / c
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG_EXTS and f.stat().st_size > 0:
                    per[s][c].append(f)
    return per


def load_labels(lbl_root: Path) -> Dict[str, Dict[str, list]]:
    per: Dict[str, Dict[str, list]] = {s: {c: [] for c in CLASSES} for s in SPLITS}
    for s in SPLITS:
        for c in CLASSES:
            p = lbl_root / s / f"{s}_{c}.json"
            if not p.is_file():
                continue
            # euc-kr 우선, 실패하면 utf-8 fallback, 둘 다 실패하면 건너뛰고 경고
            loaded = None
            last_err: Exception | None = None
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


def load_seg(seg_root: Path, split: str, cls: str):
    p = seg_root / split / f"{split}_{cls}.npz"
    if not p.is_file():
        return None
    try:
        return np.load(p, allow_pickle=True)
    except (OSError, ValueError, EOFError) as e:
        print(f"[warn] {p.name} npz 로드 실패: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 이미지 해상도/모드 샘플링 (깨진 이미지 포함 카운트)
# ──────────────────────────────────────────────────────────────
def sample_image_meta(per_img, per_class_max: int) -> Tuple[list, list, list, list]:
    all_sizes: List[Tuple[int, int]] = []
    modes: List[str] = []
    broken: List[Tuple[Path, str]] = []
    sampled_paths: List[Path] = []
    for s in SPLITS:
        for c in CLASSES:
            files = per_img[s][c]
            if not files:
                continue
            if len(files) > per_class_max:
                idx = np.linspace(0, len(files) - 1, per_class_max).astype(int)
                files = [files[i] for i in idx]
            for f in files:
                try:
                    with Image.open(f) as im:
                        # NOTE: exif_transpose — 원본 data/ 는 실제 회전 수행,
                        # data_rot/ 는 정규화로 EXIF=1 이므로 no-op (이중 회전 없음).
                        im = ImageOps.exif_transpose(im)
                        im.load()  # lazy 디코드 강제 → 깨진 이미지 탐지
                        all_sizes.append(im.size)
                        modes.append(im.mode)
                        sampled_paths.append(f)
                except (UnidentifiedImageError, OSError, SyntaxError) as e:
                    broken.append((f, str(e)))
    return all_sizes, modes, broken, sampled_paths


# ──────────────────────────────────────────────────────────────
# seg npz key 매칭 유틸 + 진단
# ──────────────────────────────────────────────────────────────
def _npz_key_for(npz_files, fname: str) -> str | None:
    """npz.files 가운데 fname(이미지 파일명)에 대응하는 key 리턴."""
    if fname in npz_files:
        return fname
    if f"{fname}.npy" in npz_files:
        return f"{fname}.npy"
    # 확장자 교체 패턴(대소문자 등)도 혹시몰라 시도
    stem = Path(fname).stem
    for k in npz_files:
        if k.startswith(stem + "."):
            return k
    return None


# ──────────────────────────────────────────────────────────────
# 1) 분포 차트
# ──────────────────────────────────────────────────────────────
def plot_distributions(per_img, per_lbl, all_sizes, modes, outpath: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) 클래스별 이미지 수
    ax = axes[0, 0]
    x = np.arange(len(CLASSES))
    w = 0.4
    n_tr = [len(per_img["train"][c]) for c in CLASSES]
    n_vl = [len(per_img["val"][c]) for c in CLASSES]
    ax.bar(x - w / 2, n_tr, w, label=f"train ({sum(n_tr)})", color="#4C72B0")
    ax.bar(x + w / 2, n_vl, w, label=f"val ({sum(n_vl)})", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_title("Class Distribution (images)")
    ymax = max([*n_tr, *n_vl, 1]) * 1.15
    ax.set_ylim(0, ymax)
    for i, (a, b) in enumerate(zip(n_tr, n_vl)):
        ax.text(i - w / 2, a + ymax * 0.01, str(a), ha="center", fontsize=9)
        ax.text(i + w / 2, b + ymax * 0.01, str(b), ha="center", fontsize=9)
    ax.legend()

    # (0,1) 해상도 2D 히스토그램 (scatter 오버플롯 대체)
    ax = axes[0, 1]
    if all_sizes:
        W = np.array([s[0] for s in all_sizes])
        H = np.array([s[1] for s in all_sizes])
        h = ax.hist2d(W, H, bins=40, cmap="viridis", cmin=1)
        plt.colorbar(h[3], ax=ax, fraction=0.04)
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title(f"Image Resolution hist2d ({len(all_sizes)} sampled)")
        ax.axvline(224, ls="--", color="red", lw=0.8)
        ax.axhline(224, ls="--", color="red", lw=0.8)
    else:
        ax.set_title("Resolution — no data")
        ax.axis("off")

    # (0,2) 종횡비 + 모드 분포
    ax = axes[0, 2]
    if all_sizes:
        ratios = np.array([s[0] / s[1] for s in all_sizes])
        ax.hist(ratios, bins=40, color="#8172B3", edgecolor="black", lw=0.3)
        ax.set_title("Aspect Ratio (W/H)")
        ax.set_xlabel("W/H")
        ax.axvline(1.0, ls="--", color="black", lw=0.8, label="square")
        ax.axvline(float(np.median(ratios)), ls="--", color="red", lw=0.8,
                   label=f"median={np.median(ratios):.2f}")
        # 오른쪽 위 모서리에 모드 카운트 텍스트
        if modes:
            mc = Counter(modes).most_common()
            txt = "\n".join(f"mode {m}: {n}" for m, n in mc[:4])
            ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.set_title("Aspect Ratio — no data")
        ax.axis("off")

    # (1,0) 연령 분포
    ax = axes[1, 0]
    ages = [int(item["age"]) for c in CLASSES for item in per_lbl["train"].get(c, [])
            if isinstance(item, dict) and isinstance(item.get("age"), (int, float))
            and 0 < item["age"] < 120]
    if ages:
        ax.hist(ages, bins=np.arange(0, 100, 5), color="#CCB974", edgecolor="black", lw=0.3)
        ax.set_title(f"Age Distribution (train, n={len(ages)})")
        ax.set_xlabel("age")
        ax.axvline(float(np.mean(ages)), ls="--", color="red", lw=0.8,
                   label=f"mean={np.mean(ages):.1f}")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Age — no data")
        ax.axis("off")

    # (1,1) 성별 × 클래스 (stacked)
    ax = axes[1, 1]
    gender_per_cls = {c: Counter(item.get("gender", "?") for item in per_lbl["train"].get(c, [])
                                 if isinstance(item, dict)) for c in CLASSES}
    gender_keys = sorted({g for ct in gender_per_cls.values() for g in ct})
    if gender_keys:
        bottom = np.zeros(len(CLASSES))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
        for i, g in enumerate(gender_keys):
            vals = np.array([gender_per_cls[c].get(g, 0) for c in CLASSES])
            ax.bar(CLASSES, vals, bottom=bottom, label=str(g), color=colors[i % len(colors)])
            bottom += vals
        ax.set_title("Gender per Class (train)")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Gender — no data")
        ax.axis("off")

    # (1,2) faceExp_uploader × 폴더 클래스 매트릭스 (한글→fallback 매핑)
    ax = axes[1, 2]
    mtx: Dict[str, Counter] = defaultdict(Counter)
    raw_keys = set()
    for c in CLASSES:
        for item in per_lbl["train"].get(c, []):
            if not isinstance(item, dict):
                continue
            fe = item.get("faceExp_uploader", "?")
            mtx[c][fe] += 1
            raw_keys.add(fe)
    raw_keys = sorted(raw_keys)
    if raw_keys:
        display_labels = (raw_keys if _HANGUL_OK
                          else [FACEEXP_FALLBACK.get(k, k) for k in raw_keys])
        mat = np.array([[mtx[c].get(k, 0) for k in raw_keys] for c in CLASSES])
        im = ax.imshow(mat, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(raw_keys)))
        ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES)
        mmax = mat.max() if mat.size else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            color="white" if v > mmax * 0.5 else "black", fontsize=8)
        ax.set_title("faceExp_uploader x folder class")
        plt.colorbar(im, ax=ax, fraction=0.04)
    else:
        ax.set_title("faceExp — no data")
        ax.axis("off")

    fig.suptitle("Emotion Dataset — EDA Report", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 2) 샘플 이미지 + bbox
# ──────────────────────────────────────────────────────────────
def plot_samples_grid(per_img, per_lbl, outpath: Path, per_class: int):
    fig, axes = plt.subplots(len(CLASSES), per_class,
                             figsize=(per_class * 3.2, len(CLASSES) * 3.2))
    # filename → label item
    lbl_map = {c: {it.get("filename", ""): it for it in per_lbl["train"].get(c, [])
                   if isinstance(it, dict)} for c in CLASSES}
    rng = np.random.default_rng(42)
    for r, c in enumerate(CLASSES):
        files = per_img["train"][c]
        n_draw = min(per_class, len(files))
        if n_draw == 0:
            for k in range(per_class):
                axes[r, k].axis("off")
                axes[r, k].set_title(f"{c} — no image", fontsize=9)
            continue
        idx = rng.choice(len(files), size=n_draw, replace=False)
        for k, i in enumerate(idx):
            ax = axes[r, k]
            f = files[i]
            try:
                with Image.open(f) as im:
                    # NOTE: exif_transpose — data/ 는 실제 회전, data_rot/ 는 EXIF=1 로 no-op.
                    im = ImageOps.exif_transpose(im)
                    im.load()
                    ax.imshow(im)
                    w0, h0 = im.size
                info = lbl_map[c].get(f.name)
                if info and isinstance(info.get("annot_A"), dict):
                    box = info["annot_A"].get("boxes", {})
                    if all(k in box for k in ("minX", "maxX", "minY", "maxY")):
                        mn_x, mx_x = box["minX"], box["maxX"]
                        mn_y, mx_y = box["minY"], box["maxY"]
                        ax.add_patch(plt.Rectangle((mn_x, mn_y), mx_x - mn_x, mx_y - mn_y,
                                                   fill=False, ec="red", lw=2))
                ax.set_title(f"{c} / {w0}x{h0}", fontsize=9)
            except Exception as e:
                ax.set_title(f"err: {type(e).__name__}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        # 남는 컬럼(이미지 부족) 가드
        for k in range(n_draw, per_class):
            axes[r, k].axis("off")

    fig.suptitle("Samples per class (train) with annot_A bbox", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 3) segmentation mask 오버레이
# ──────────────────────────────────────────────────────────────
SEG_PALETTE = np.array([
    [0, 0, 0],        # 0 bg
    [80, 80, 255],    # 1 hair
    [0, 180, 120],    # 2 body
    [255, 120, 90],   # 3 face
    [240, 210, 60],   # 4 cloth
    [180, 180, 180],  # 5 etc
], dtype=np.uint8)


def plot_seg_overlay(per_img, seg_root: Path, outpath: Path):
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(12, len(CLASSES) * 3.2))
    rng = np.random.default_rng(7)
    for r, c in enumerate(CLASSES):
        npz = load_seg(seg_root, "train", c)
        files = per_img["train"][c]
        if npz is None or not files:
            for k in range(3):
                axes[r, k].axis("off")
            axes[r, 0].set_title(f"{c} — no seg/img", fontsize=9)
            continue

        npz_files = set(npz.files)

        # 진단: 첫 클래스일 때 실제 key 샘플 출력
        if r == 0:
            print(f"[seg-debug] class={c} npz key 샘플: {list(npz_files)[:3]}")
            print(f"[seg-debug] 이미지 파일명 샘플: {files[0].name if files else 'NONE'}")

        # 매칭되는 파일 우선 선택
        matched = [f for f in files if _npz_key_for(npz_files, f.name)]
        pool = matched if matched else files
        f = pool[int(rng.integers(0, len(pool)))]

        # 이미지
        try:
            with Image.open(f) as im0:
                # NOTE: exif_transpose — data/ 는 실제 회전, data_rot/ 는 EXIF=1 로 no-op.
                im0 = ImageOps.exif_transpose(im0)
                im0.load()
                img_arr = np.array(im0.convert("RGB"))
        except Exception as e:
            for k in range(3):
                axes[r, k].axis("off")
            axes[r, 0].set_title(f"{c} — img err: {type(e).__name__}", fontsize=9)
            npz.close()  # 파일 핸들 누수 방지
            continue

        # 마스크
        key = _npz_key_for(npz_files, f.name)
        mask = None
        if key is not None:
            try:
                mask = npz[key]
            except (KeyError, OSError, ValueError) as e:
                print(f"[warn] {f.name} mask 읽기 실패: {e}")
                mask = None

        axes[r, 0].imshow(img_arr)
        axes[r, 0].set_title(f"{c} — original", fontsize=10)
        axes[r, 0].axis("off")

        if mask is not None:
            mh, mw = mask.shape[:2]
            # 이미지와 마스크 크기 다르면 이미지를 마스크 크기로 맞춤 (RGB 보장)
            if (mh, mw) != img_arr.shape[:2]:
                img_small = np.array(
                    Image.fromarray(img_arr).resize((mw, mh)).convert("RGB")
                )
            else:
                img_small = img_arr
            color = SEG_PALETTE[np.clip(mask, 0, 5).astype(np.int32)]
            overlay = (0.5 * img_small + 0.5 * color).astype(np.uint8)
            axes[r, 1].imshow(color)
            axes[r, 1].set_title("mask (palette)", fontsize=10)
            axes[r, 1].axis("off")
            axes[r, 2].imshow(overlay)
            uniq = np.unique(mask).tolist()
            axes[r, 2].set_title(f"overlay — vals={uniq}", fontsize=10)
            axes[r, 2].axis("off")
        else:
            axes[r, 1].axis("off")
            axes[r, 2].axis("off")
            axes[r, 1].set_title("(mask missing)", fontsize=9)

        # 파일 핸들 누수 방지 (np.load 는 lazy; close 필수)
        npz.close()

    fig.suptitle("Segmentation mask overlay (0=bg 1=hair 2=body 3=face 4=cloth 5=etc)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 4) bbox 분석
# ──────────────────────────────────────────────────────────────
def plot_bbox_analysis(per_lbl, outpath: Path):
    ws: Dict[str, list] = defaultdict(list)
    hs: Dict[str, list] = defaultdict(list)
    for c in CLASSES:
        for item in per_lbl["train"].get(c, []):
            if not isinstance(item, dict):
                continue
            annot = item.get("annot_A", {})
            b = annot.get("boxes", {}) if isinstance(annot, dict) else {}
            try:
                bw = b["maxX"] - b["minX"]
                bh = b["maxY"] - b["minY"]
            except (KeyError, TypeError):
                continue
            if bw > 0 and bh > 0:
                ws[c].append(float(bw))
                hs[c].append(float(bh))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cls_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    def _boxplot(ax, data, title, ylabel):
        if not any(data):
            ax.axis("off"); ax.set_title(title + " — no data"); return
        # matplotlib 3.9+: tick_labels; 이전: labels — 둘 다 대응
        try:
            bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(data, labels=CLASSES, patch_artist=True)
        for patch, color in zip(bp["boxes"], cls_colors):
            patch.set_facecolor(color)
        ax.set_title(title); ax.set_ylabel(ylabel)

    _boxplot(axes[0], [ws[c] for c in CLASSES], "bbox width by class (px)", "width (px)")
    _boxplot(axes[1], [hs[c] for c in CLASSES], "bbox height by class (px)", "height (px)")

    ax = axes[2]
    plotted = False
    for color, c in zip(cls_colors, CLASSES):
        if ws[c] and hs[c]:
            ar = np.array(ws[c]) / np.array(hs[c])
            ax.hist(ar, bins=30, alpha=0.5, label=c, color=color)
            plotted = True
    if plotted:
        ax.set_title("bbox aspect ratio (W/H) by class")
        ax.legend(fontsize=8)
        ax.axvline(1.0, ls="--", color="black", lw=0.6)
    else:
        ax.axis("off"); ax.set_title("bbox AR — no data")

    fig.suptitle("annot_A bbox analysis (train)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 5) 마크다운 리포트
# ──────────────────────────────────────────────────────────────
def write_summary_md(per_img, per_lbl, all_sizes, modes, broken, seg_root: Path, outpath: Path):
    lines = ["# EDA 요약 — 감정 분류 프로젝트", ""]
    n_total = sum(len(per_img[s][c]) for s in SPLITS for c in CLASSES)
    n_tr = sum(len(per_img["train"][c]) for c in CLASSES)
    n_vl = sum(len(per_img["val"][c]) for c in CLASSES)
    lines += [
        "## 전체 수량",
        f"- 이미지 총: **{n_total}** (train {n_tr} / val {n_vl})",
        f"- 깨진/열기 실패: {len(broken)} (샘플링 기준)",
        "",
        "## 클래스 분포",
        "",
        "| 클래스 | train | val | 합계 |",
        "|---|---:|---:|---:|",
    ]
    for c in CLASSES:
        t = len(per_img["train"][c])
        v = len(per_img["val"][c])
        lines.append(f"| {c} ({CLASS_KOR[c]}) | {t} | {v} | {t + v} |")
    lines.append("")

    if all_sizes:
        Ws = np.array([s[0] for s in all_sizes])
        Hs = np.array([s[1] for s in all_sizes])
        lines += [
            "## 이미지 해상도 (샘플링)",
            f"- 너비  W : min {int(Ws.min())}, median {int(np.median(Ws))}, max {int(Ws.max())}",
            f"- 높이  H : min {int(Hs.min())}, median {int(np.median(Hs))}, max {int(Hs.max())}",
            f"- 종횡비 W/H: median {np.median(Ws / Hs):.2f}  ({(Ws / Hs).min():.2f} ~ {(Ws / Hs).max():.2f})",
            f"- 224×224 resize 대상: 원본 대체로 고해상도 → 다운샘플링 손실 주의",
            "",
        ]
    if modes:
        mc = Counter(modes).most_common()
        lines += ["## 이미지 채널 모드 (PIL `Image.mode`)"]
        for m, n in mc:
            lines.append(f"- `{m}` : {n}")
        lines.append("")

    # 매칭
    lines += ["## 이미지 ↔ 라벨 매칭", "",
              "| split/class | 이미지 | 라벨 | 매칭 | img전용 | lbl전용 |",
              "|---|---:|---:|---:|---:|---:|"]
    total_mismatch = 0
    for s in SPLITS:
        for c in CLASSES:
            img_set = {f.name for f in per_img[s][c]}
            items = per_lbl[s].get(c, [])
            lbl_set = {it.get("filename", "") for it in items if isinstance(it, dict)}
            matched = img_set & lbl_set
            only_i = img_set - lbl_set
            only_l = lbl_set - img_set
            total_mismatch += len(only_i) + len(only_l)
            lines.append(f"| {s}/{c} | {len(img_set)} | {len(lbl_set)} | {len(matched)} | {len(only_i)} | {len(only_l)} |")
    lines += ["", f"- 매칭 불일치 총: **{total_mismatch}**", ""]

    # 라벨 부가 정보
    lines += ["## 라벨 부가 정보 (train)"]
    ages = [int(item["age"]) for c in CLASSES for item in per_lbl["train"].get(c, [])
            if isinstance(item, dict) and isinstance(item.get("age"), (int, float))
            and 0 < item["age"] < 120]
    if ages:
        lines.append(f"- 연령 n={len(ages)}: mean {np.mean(ages):.1f}, median {int(np.median(ages))}, min {min(ages)}, max {max(ages)}")
    gender_ct = Counter(item.get("gender", "?") for c in CLASSES
                        for item in per_lbl["train"].get(c, []) if isinstance(item, dict))
    if gender_ct:
        lines.append("- 성별: " + ", ".join(f"{k}={v}" for k, v in gender_ct.most_common()))
    isprof_ct = Counter(item.get("isProf", "?") for c in CLASSES
                        for item in per_lbl["train"].get(c, []) if isinstance(item, dict))
    if isprof_ct:
        lines.append("- isProf: " + ", ".join(f"{k}={v}" for k, v in isprof_ct.most_common()))
    lines.append("")

    # Segmentation 체크
    lines += ["## Segmentation 요약", ""]
    for s in SPLITS:
        for c in CLASSES:
            npz = load_seg(seg_root, s, c)
            if npz is None:
                lines.append(f"- {s}/{c}: 파일 없음 또는 로드 실패")
                continue
            try:
                if not npz.files:
                    lines.append(f"- {s}/{c}: 빈 npz")
                    continue
                first = npz.files[0]
                try:
                    arr = npz[first]
                    shape = arr.shape
                    uniq = np.unique(arr).tolist()
                    lines.append(f"- {s}/{c}: {len(npz.files)} masks, sample shape={shape}, vals={uniq}")
                except Exception as e:
                    lines.append(f"- {s}/{c}: 첫 mask 읽기 실패 ({e})")
            finally:
                # 파일 핸들 누수 방지 (np.load 는 lazy; close 필수)
                npz.close()
    lines.append("")

    lines += [
        "## 산출물",
        "- `results/eda_report.png` — 분포 차트 6개",
        "- `results/samples_grid.png` — 클래스별 샘플 + bbox 오버레이",
        "- `results/seg_overlay.png` — segmentation 마스크 시각화",
        "- `results/bbox_analysis.png` — bbox 크기/종횡비 박스플롯",
        "",
        "## 판단 포인트",
        "- 클래스 거의 균등 → class_weight 큰 조정 불필요",
        "- 원본 해상도 편차 큼 → 224 resize 시 정보 손실 → face crop(annot_A) 필수",
        "- seg 마스크로 배경/옷 제거 실험 → 창의성 + 모델링 점수 공략",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    root = Path(args.root)
    data = root / args.data_subdir
    img_root = data / "img"
    lbl_root = data / "label"
    seg_root = data / "segmentation"
    out = root / args.results_subdir
    out.mkdir(exist_ok=True)
    tag = args.tag

    if not img_root.is_dir():
        print(f"[!] {img_root} 없음. --root 또는 $PROJECT_ROOT 확인")
        sys.exit(2)

    print(f"[1/5] scanning images + labels @ {root}")
    per_img = scan_images(img_root)
    per_lbl = load_labels(lbl_root)

    print(f"[2/5] sampling image meta (size, mode) — 클래스당 최대 {args.size_sample}")
    all_sizes, modes, broken, _ = sample_image_meta(per_img, args.size_sample)

    print("[3/5] plot_distributions ...")
    plot_distributions(per_img, per_lbl, all_sizes, modes, out / f"{tag}eda_report.png")

    print("[4/5] samples + seg + bbox ...")
    plot_samples_grid(per_img, per_lbl, out / f"{tag}samples_grid.png", args.samples_per_class)
    plot_seg_overlay(per_img, seg_root, out / f"{tag}seg_overlay.png")
    plot_bbox_analysis(per_lbl, out / f"{tag}bbox_analysis.png")

    print("[5/5] summary md ...")
    write_summary_md(per_img, per_lbl, all_sizes, modes, broken, seg_root, out / f"{tag}eda_summary.md")

    # 콘솔 요약
    n_total = sum(len(per_img[s][c]) for s in SPLITS for c in CLASSES)
    print("\n=== EDA 완료 ===")
    print(f"총 이미지: {n_total}")
    for c in CLASSES:
        t = len(per_img["train"][c])
        v = len(per_img["val"][c])
        print(f"  {c}: train={t}, val={v}")
    print(f"깨진 이미지(샘플링 범위): {len(broken)}")
    print(f"산출물: {out}")
    for f in sorted(out.iterdir()):
        print(f"  - {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
