"""eval_metrics_full.py — 앙상블 config 에 대해 full metrics 패널 계산.

GPT 분석이 지적한 부족한 지표 추가:
  - Top-1 Accuracy (이미 있음)
  - Macro F1 (이미 있음)
  - **Top-2 Accuracy** (신규)
  - **ROC-AUC (OvR macro)** (신규)
  - **Cohen's Kappa** (신규)
  - Confusion Matrix
  - per-class Precision/Recall/F1
  - NLL (이미 있음)
  - Brier score (multi-class 분해)
  - ECE (Expected Calibration Error, 10 bins)

앙상블 config (json) + ensemble_cache_phase2/ 의 per-model probs npz 기반.
순수 CPU 작업 → GPU 경쟁 없음 (exp12 학습 중에도 안전).

실행:
  python scripts/eval_metrics_full.py \
    --config models/ensemble_fc_I_5m_exp11.json \
    --cache-dir results/ensemble_cache_phase2 \
    --out results/metrics_ensemble_I.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT = Path(os.environ.get(
    "EMOTION_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import predict as predict_mod  # noqa: E402

CLASSES: List[str] = list(predict_mod.CLASSES)
NUM_CLASSES: int = predict_mod.NUM_CLASSES
EPS = 1e-8


# --------------------------------------------------------------------
# probs 로드
# --------------------------------------------------------------------
def _model_cache_key(model_path: Path) -> str:
    st = model_path.stat() if model_path.is_file() else None
    mtime_ns = st.st_mtime_ns if st else 0
    return f"{model_path.stem}.mtime{mtime_ns}"


def load_member_probs(model_path: Path, cache_dir: Path, crop_mode: str = "bbox",
                      expected_n: int = -1):
    """cache 에서 probs 로드. expected_n 주어지면 길이 일치 안 하는 cache 는 무시 (강제 re-infer)."""
    key = _model_cache_key(model_path)
    cache_file = cache_dir / f"{key}_{crop_mode}_val_probs.npz"
    if not cache_file.is_file():
        # 이름 매칭 완화: 같은 stem 의 아무 mtime 이나
        stem = model_path.stem
        candidates = sorted(cache_dir.glob(f"{stem}.mtime*_{crop_mode}_val_probs.npz"))
        if not candidates:
            raise FileNotFoundError(f"cache 없음: {model_path.name} (stem={stem}, crop={crop_mode})")
        cache_file = candidates[-1]  # 가장 최신 mtime
    dat = np.load(cache_file, allow_pickle=False)
    probs = dat["probs"].astype(np.float64)
    # 길이 sanity check — val_dir 가 다르면 cache 길이가 안 맞을 수 있음
    if expected_n >= 0 and probs.shape[0] != expected_n:
        raise FileNotFoundError(
            f"cache 길이 불일치: {model_path.name} cached={probs.shape[0]} vs expected={expected_n}. "
            f"새 val_dir 일 가능성 → 직접 추론으로 강제 fallback."
        )
    return probs, dat.get("paths", None)


def infer_member_probs(model_path: Path, paths: list, crop_mode: str = "bbox"):
    """cache 없을 때 predict.py 로 직접 추론. cache_dir 에 저장도 함."""
    import time as _t
    from PIL import Image, ImageOps
    import predict as predict_mod  # noqa: E402

    print(f"  [infer] {model_path.name} 직접 추론 (cache 없음, 첫 실행)", file=sys.stderr)
    m_obj = predict_mod.load_model(model_path)
    n = len(paths)
    probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    t0 = _t.perf_counter()
    for i, p in enumerate(paths):
        pil = Image.open(p).convert("RGB")
        pil = ImageOps.exif_transpose(pil)
        pr = predict_mod.predict_probs(m_obj, pil)
        probs[i] = pr.astype(np.float64)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            dt = _t.perf_counter() - t0
            print(f"    [{i+1}/{n}] {dt:.0f}s", file=sys.stderr)
    return probs


def load_or_infer_member_probs(model_path: Path, cache_dir: Path,
                               paths: list, crop_mode: str = "bbox"):
    """cache hit 시 npz 로드 (길이 검증 후), miss/길이 mismatch 시 직접 추론 후 cache 저장."""
    try:
        return load_member_probs(model_path, cache_dir, crop_mode=crop_mode,
                                 expected_n=len(paths))
    except FileNotFoundError as e:
        if "불일치" in str(e):
            print(f"  [warn] {e}", file=sys.stderr)
        probs = infer_member_probs(model_path, paths, crop_mode=crop_mode)
        # 이번 결과를 cache 에 저장 — 다음 실행 빠르게
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = _model_cache_key(model_path)
            cache_file = cache_dir / f"{key}_{crop_mode}_val_probs.npz"
            np.savez_compressed(cache_file, probs=probs.astype(np.float32))
            print(f"  [cache] saved → {cache_file.name}", file=sys.stderr)
        except Exception as e:
            print(f"  [cache] save 실패 (무시): {e}", file=sys.stderr)
        return probs, None


def load_val_labels(val_dir: Path):
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths, y = [], []
    for ci, cls in enumerate(CLASSES):
        for p in sorted((val_dir / cls).iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                paths.append(str(p))
                y.append(ci)
    return np.array(y, dtype=np.int64), paths


# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------
def compute_all_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """probs: (N, C) softmax, y: (N,) int."""
    N, C = probs.shape
    assert C == NUM_CLASSES

    pred = np.argmax(probs, axis=1)
    # Top-1
    top1 = float((pred == y).mean())

    # Top-2
    top2_idx = np.argsort(-probs, axis=1)[:, :2]
    top2 = float((top2_idx == y[:, None]).any(axis=1).mean())

    # NLL
    p_true = probs[np.arange(N), y]
    nll = float(-np.log(np.clip(p_true, EPS, 1.0)).mean())

    # Macro F1 + per-class
    per_class = {}
    f1s, precs, recs = [], [], []
    for c in range(C):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        per_class[CLASSES[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int((y == c).sum()),
        }
        f1s.append(f1); precs.append(prec); recs.append(rec)
    macro_f1 = float(np.mean(f1s))
    macro_prec = float(np.mean(precs))
    macro_rec = float(np.mean(recs))

    # Confusion Matrix
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[int(t), int(p)] += 1

    # Cohen's Kappa (통계 유의성 척도)
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = float(cohen_kappa_score(y, pred))
    except Exception as e:
        kappa = None
        print(f"[warn] cohen_kappa 실패: {e}", file=sys.stderr)

    # ROC-AUC (OvR macro — multi-class 용)
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc_macro = float(roc_auc_score(
            y, probs, multi_class="ovr", average="macro", labels=list(range(C))))
        roc_auc_weighted = float(roc_auc_score(
            y, probs, multi_class="ovr", average="weighted", labels=list(range(C))))
    except Exception as e:
        roc_auc_macro = None
        roc_auc_weighted = None
        print(f"[warn] roc_auc 실패: {e}", file=sys.stderr)

    # Brier (multi-class)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(N), y] = 1.0
    brier = float(((probs - y_onehot) ** 2).sum(axis=1).mean())

    # ECE (Expected Calibration Error, 10 bins)
    conf = probs.max(axis=1)
    correct = (pred == y).astype(np.float64)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / N) * abs(bin_conf - bin_acc)

    return {
        "n_samples": int(N),
        "top1_accuracy": round(top1, 4),
        "top2_accuracy": round(top2, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "roc_auc_macro_ovr": round(roc_auc_macro, 4) if roc_auc_macro is not None else None,
        "roc_auc_weighted_ovr": round(roc_auc_weighted, 4) if roc_auc_weighted is not None else None,
        "nll": round(nll, 4),
        "brier_multiclass": round(brier, 4),
        "ece_10bin": round(float(ece), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": list(CLASSES),
    }


def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> dict:
    """paired McNemar test — A 와 B 중 누가 통계적으로 더 정확한지."""
    a_correct = (pred_a == y)
    b_correct = (pred_b == y)
    n01 = int(((~a_correct) & b_correct).sum())  # A 틀리고 B 맞춤
    n10 = int((a_correct & (~b_correct)).sum())  # A 맞고 B 틀림
    # continuity correction McNemar
    if n01 + n10 == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
        # chi2 df=1 → p
        from math import erfc, sqrt
        p = erfc(sqrt(chi2) / sqrt(2))
    return {
        "n_a_only_correct": n10,
        "n_b_only_correct": n01,
        "chi2": round(chi2, 4),
        "p_value": round(float(p), 4),
        "significant_at_0.05": bool(p < 0.05),
    }


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="앙상블 config json")
    ap.add_argument("--model", default=None,
                    help="단일 모델 평가 시 .pt / .h5 / .json 한 파일 경로")
    ap.add_argument("--cache-dir", default=None,
                    help="멤버 probs 캐시 폴더 (--config 일 때 필요)")
    ap.add_argument("--val-dir", default=str(PROJECT / "data_rot/img/val"))
    ap.add_argument("--out", default=None,
                    help="지표 패널을 저장할 json 경로 (없으면 stdout 만)")
    ap.add_argument("--crop-mode", default="bbox", choices=["bbox", "raw"])
    ap.add_argument("--compare-config", default=None,
                    help="McNemar test 대상: 두 번째 config (e.g. 구 앙상블)")
    args = ap.parse_args()

    # --model 과 --config 둘 다 없으면 에러, 둘 다면 --config 우선
    if not args.config and not args.model:
        ap.error("--config 또는 --model 중 하나는 필수")
    if args.model and not args.config:
        # --model 이 .json 이면 --config 로 자동 매핑 (사용자 편의)
        if str(args.model).endswith(".json"):
            args.config = args.model
            args.model = None

    # 1) val labels
    val_dir = Path(args.val_dir)
    y, paths = load_val_labels(val_dir)
    print(f"[val] n={len(y)}  val_dir={val_dir}", file=sys.stderr)

    # 단일 모델 분기 — predict.py 직접 호출로 probs 만 뽑고 metrics 계산
    if args.model and not args.config:
        single_path = Path(args.model)
        if not single_path.is_absolute():
            single_path = PROJECT / single_path
        if not single_path.is_file():
            ap.error(f"--model 파일 없음: {single_path}")
        print(f"[single] {single_path.name}", file=sys.stderr)

        import predict as predict_mod  # noqa: E402
        m_obj = predict_mod.load_model(single_path)
        probs_single = np.zeros((len(paths), NUM_CLASSES), dtype=np.float64)
        from PIL import Image, ImageOps
        import time as _t
        t0 = _t.perf_counter()
        for i, p in enumerate(paths):
            pil = Image.open(p).convert("RGB")
            pil = ImageOps.exif_transpose(pil)
            pr = predict_mod.predict_probs(m_obj, pil)
            probs_single[i] = pr.astype(np.float64)
            if (i + 1) % 200 == 0 or (i + 1) == len(paths):
                dt = _t.perf_counter() - t0
                print(f"  [{i+1}/{len(paths)}] {dt:.1f}s", file=sys.stderr)

        metrics_s = compute_all_metrics(probs_single, y)
        metrics_s["model"] = str(single_path)
        metrics_s["crop_mode"] = "predict.py default (auto bbox/MTCNN)"

        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(metrics_s, f, indent=2, ensure_ascii=False)
            print(f"\n[saved] {out}", file=sys.stderr)

        # stdout 요약
        print("\n" + "=" * 60)
        print(f"Single Model Metrics — {single_path.name}")
        print("=" * 60)
        for k in ("top1_accuracy", "top2_accuracy", "macro_f1", "macro_precision",
                  "macro_recall", "cohen_kappa", "roc_auc_macro_ovr",
                  "roc_auc_weighted_ovr", "nll", "brier_multiclass", "ece_10bin"):
            v = metrics_s.get(k)
            print(f"  {k:<30s}  {v}")
        print("\nPer-class (F1):")
        for cls, r in metrics_s["per_class"].items():
            print(f"  {cls:<8s}  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  n={r['support']}")
        return

    if not args.cache_dir:
        ap.error("--config 사용 시 --cache-dir 필수")
    if not args.out:
        ap.error("--config 사용 시 --out 필수")

    # 2) config 로드 + 멤버 probs 로드 + weighted avg
    cfg = json.loads(Path(args.config).read_text())
    models = cfg["models"]
    print(f"[config] {args.config}  M={len(models)}", file=sys.stderr)

    cache_dir = Path(args.cache_dir)
    probs_list, weights, names = [], [], []
    failed_members = []
    for m in models:
        mpath = Path(m["path"])
        if not mpath.is_absolute():
            mpath = PROJECT / mpath
        # symlink 타고 실제 파일로
        if mpath.is_symlink():
            mpath = Path(os.path.realpath(mpath))
        try:
            p, _ = load_or_infer_member_probs(mpath, cache_dir, paths=paths,
                                              crop_mode=args.crop_mode)
            probs_list.append(p)
            weights.append(float(m["weight"]))
            names.append(Path(m["path"]).name)
            print(f"  [{Path(m['path']).name}] w={m['weight']:.4f}", file=sys.stderr)
        except Exception as e:
            failed_members.append((Path(m["path"]).name, str(e)[:200]))
            print(f"  [SKIP] {Path(m['path']).name} 로드 실패 — {str(e)[:120]}", file=sys.stderr)

    if not probs_list:
        ap.error(f"모든 멤버 로드 실패. 환경 (TF/PyTorch 설치) 확인 필요. 실패 목록: {failed_members}")
    if failed_members:
        print(f"\n[warn] 멤버 {len(failed_members)} 개 로드 실패 → 남은 {len(probs_list)} 개로 앙상블 진행. "
              f"weight 재정규화됨.\n        실패: {[n for n, _ in failed_members]}\n",
              file=sys.stderr)

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    # weighted avg
    stacked = np.stack(probs_list, axis=0)  # (M, N, C)
    probs_ens = (stacked * weights[:, None, None]).sum(axis=0)
    # 정규화 (already probs but safe)
    probs_ens = probs_ens / probs_ens.sum(axis=1, keepdims=True)

    # 3) metrics
    metrics = compute_all_metrics(probs_ens, y)
    metrics["config"] = str(args.config)
    metrics["crop_mode"] = args.crop_mode

    # 4) 개별 모델 metric 도 추가 (참고)
    per_member = {}
    for m, p in zip(models, probs_list):
        name = Path(m["path"]).stem
        per_member[name] = compute_all_metrics(p, y)
    metrics["per_member"] = per_member

    # 5) McNemar test (compare-config 있으면)
    if args.compare_config:
        cfg2 = json.loads(Path(args.compare_config).read_text())
        probs_list2, weights2 = [], []
        for m in cfg2["models"]:
            mpath = Path(m["path"])
            if not mpath.is_absolute():
                mpath = PROJECT / mpath
            if mpath.is_symlink():
                mpath = Path(os.path.realpath(mpath))
            try:
                p, _ = load_member_probs(mpath, cache_dir, crop_mode=args.crop_mode)
                probs_list2.append(p)
                weights2.append(float(m["weight"]))
            except FileNotFoundError as e:
                print(f"[warn] compare cache 없음: {e}", file=sys.stderr)
                probs_list2 = None
                break

        if probs_list2:
            w2 = np.array(weights2); w2 = w2 / w2.sum()
            probs2_ens = (np.stack(probs_list2, axis=0) * w2[:, None, None]).sum(axis=0)
            probs2_ens = probs2_ens / probs2_ens.sum(axis=1, keepdims=True)
            pred1 = np.argmax(probs_ens, axis=1)
            pred2 = np.argmax(probs2_ens, axis=1)
            mc = mcnemar_test(pred1, pred2, y)
            mc["config_a"] = str(args.config)
            mc["config_b"] = str(args.compare_config)
            metrics["mcnemar_vs_compare"] = mc

    # 6) save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}", file=sys.stderr)

    # 7) 요약 표 stdout
    print("\n" + "=" * 60)
    print(f"Ensemble Metrics Panel — {Path(args.config).name}")
    print("=" * 60)
    for k in ("top1_accuracy", "top2_accuracy", "macro_f1", "macro_precision",
              "macro_recall", "cohen_kappa", "roc_auc_macro_ovr",
              "roc_auc_weighted_ovr", "nll", "brier_multiclass", "ece_10bin"):
        v = metrics.get(k)
        print(f"  {k:<30s}  {v}")
    print("\nPer-class (F1):")
    for cls, r in metrics["per_class"].items():
        print(f"  {cls:<8s}  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  n={r['support']}")
    if "mcnemar_vs_compare" in metrics:
        mc = metrics["mcnemar_vs_compare"]
        print(f"\nMcNemar vs compare: chi2={mc['chi2']}, p={mc['p_value']}, sig@0.05={mc['significant_at_0.05']}")
        print(f"  A only correct: {mc['n_a_only_correct']}")
        print(f"  B only correct: {mc['n_b_only_correct']}")


if __name__ == "__main__":
    main()
