#!/usr/bin/env python
"""GPT-5.1 Vision (zero-shot) vs 우리 앙상블 대결.

같은 val 40장 (class 5×4 혹은 10×4, seed=42) 에 대해
GPT-5.1 Vision chat completions + image_url content 로 분류,
우리 앙상블 예측, 누가 더 맞추는지 실측.

환경변수 OPENAI_API_KEY 필수.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

PROJECT = Path("/workspace/user4/emotion-project")
sys.path.insert(0, str(PROJECT))
from predict import CLASSES, load_model, predict_probs  # noqa: E402

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")


def gpt_vision(image_path: Path, model: str = "gpt-5.1",
               timeout: int = 60) -> dict:
    """GPT Vision 한 장 분류. zero-shot, 라벨 정의 안 줌 (공정)."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a facial emotion classifier. "
                    "Output EXACTLY one word from this list: anger, happy, panic, sadness. "
                    "No explanation, no punctuation, no other text."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which emotion? Answer with one word only."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ],
        "max_completion_tokens": 20,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    reply = (body["choices"][0]["message"]["content"] or "").strip().lower()
    # 라벨 단어 스캔
    pred = None
    for c in CLASSES:
        if c in reply:
            pred = c
            break
    return {"raw": reply, "pred": pred, "dt": dt,
            "tokens": (body.get("usage") or {}).get("total_tokens")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpt-model", default="gpt-5.1")
    ap.add_argument("--output", default=str(PROJECT / "results/gpt_vs_ensemble.json"))
    args = ap.parse_args()

    if not OPENAI_KEY.startswith("sk-"):
        print("ERROR: OPENAI_API_KEY env 변수 필요", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    samples = []
    for cls in CLASSES:
        imgs = sorted((PROJECT / "data_rot/img/val" / cls).glob("*.jpg"))
        picks = rng.sample(imgs, args.n_per_class)
        for p in picks:
            samples.append((str(p), cls))
    print(f"[samples] n={len(samples)} ({args.n_per_class}/class, seed={args.seed})",
          file=sys.stderr)

    print("[load] ensemble...", file=sys.stderr)
    t0 = time.perf_counter()
    model = load_model(str(PROJECT / "models/ensemble_with_kd.json"),
                       tta=False, auto_face_crop=True)
    print(f"  load {time.perf_counter() - t0:.1f}s  type={model['type']}",
          file=sys.stderr)

    print("[warmup] 2회 (XLA compile)", file=sys.stderr)
    for i in range(2):
        _ = predict_probs(model, samples[i][0])

    print("\n" + "=" * 90)
    print(f"{'#':>3s} {'GT':8s} {'Ensemble':>10s} {'conf':>6s} | {'GPT-5.1':>10s} {'dt':>6s} | {'verdict':>10s}")
    print("=" * 90)

    ens_preds, gpt_preds, gpt_raws, gpt_times = [], [], [], []
    for i, (path, gt) in enumerate(samples, 1):
        probs = predict_probs(model, path)
        ens_pred = CLASSES[int(np.argmax(probs))]
        ens_conf = float(probs.max())
        ens_preds.append(ens_pred)

        try:
            g = gpt_vision(Path(path), model=args.gpt_model)
            gpt_pred = g["pred"] or "UNKNOWN"
            gpt_preds.append(gpt_pred)
            gpt_raws.append(g["raw"])
            gpt_times.append(g["dt"])
        except Exception as e:
            print(f"  [err] {path}: {e}", file=sys.stderr)
            gpt_preds.append("ERR")
            gpt_raws.append(str(e)[:100])
            gpt_times.append(0.0)
            continue

        em = "OK" if ens_pred == gt else "X "
        gm = "OK" if gpt_preds[-1] == gt else "X "
        if ens_pred == gt and gpt_preds[-1] == gt:
            verdict = "both-right"
        elif ens_pred == gt:
            verdict = "ens-only"
        elif gpt_preds[-1] == gt:
            verdict = "gpt-only"
        else:
            verdict = "both-wrong"
        print(f"{i:>3d} {gt:8s} {em} {ens_pred:>8s} {ens_conf:.3f} | "
              f"{gm} {gpt_preds[-1]:>8s} {gpt_times[-1]:>4.1f}s | {verdict:>10s}")

    # 집계
    ens_correct = sum(1 for p, s in zip(ens_preds, samples) if p == s[1])
    gpt_valid = [(p, s[1]) for p, s in zip(gpt_preds, samples)
                 if p not in ("ERR", "UNKNOWN")]
    gpt_correct = sum(1 for p, gt in gpt_valid if p == gt)
    n_unparseable = len(samples) - len(gpt_valid)

    ens_acc = ens_correct / len(samples)
    gpt_acc = gpt_correct / max(1, len(gpt_valid))

    print(f"\n{'='*90}")
    print(f"앙상블       : {ens_correct}/{len(samples)}  ({ens_acc*100:.1f}%)")
    print(f"GPT-{args.gpt_model}: {gpt_correct}/{len(gpt_valid)}  ({gpt_acc*100:.1f}%)"
          f"{'  (+' + str(n_unparseable) + ' unparseable)' if n_unparseable else ''}")
    print(f"격차         : {(ens_acc - gpt_acc)*100:+.1f}%p  (앙상블 기준)")
    print(f"GPT latency  : mean {sum(gpt_times)/max(1,len(gpt_times)):.1f}s/img")

    # 뒤바뀜
    ens_only = sum(1 for e, g, s in zip(ens_preds, gpt_preds, samples)
                   if e == s[1] and g != s[1])
    gpt_only = sum(1 for e, g, s in zip(ens_preds, gpt_preds, samples)
                   if e != s[1] and g == s[1] and g not in ("ERR", "UNKNOWN"))
    both_right = sum(1 for e, g, s in zip(ens_preds, gpt_preds, samples)
                     if e == s[1] and g == s[1])
    both_wrong = sum(1 for e, g, s in zip(ens_preds, gpt_preds, samples)
                     if e != s[1] and g != s[1])

    print(f"\n케이스 분석:")
    print(f"  둘 다 맞춤   : {both_right}")
    print(f"  앙상블만 맞춤: {ens_only}")
    print(f"  GPT만 맞춤   : {gpt_only}")
    print(f"  둘 다 틀림   : {both_wrong}")

    print(f"\n클래스별:")
    print(f"{'class':10s} {'앙상블':>10s} {'GPT':>10s}")
    for cls in CLASSES:
        idxs = [i for i, s in enumerate(samples) if s[1] == cls]
        e = sum(1 for i in idxs if ens_preds[i] == cls)
        g = sum(1 for i in idxs if gpt_preds[i] == cls)
        print(f"  {cls:<8s} {e}/{len(idxs)}       {g}/{len(idxs)}")

    # GPT 가 가장 많이 본 답
    from collections import Counter
    print(f"\nGPT 예측 분포 (공정 zero-shot 한계 체크):")
    for k, v in Counter(gpt_preds).most_common():
        print(f"  {k:10s} {v}")

    out = {
        "n": len(samples),
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "gpt_model": args.gpt_model,
        "ensemble": {"correct": ens_correct, "total": len(samples), "acc": ens_acc},
        "gpt": {"correct": gpt_correct, "valid": len(gpt_valid),
                "unparseable": n_unparseable, "acc": gpt_acc,
                "mean_latency_sec": sum(gpt_times) / max(1, len(gpt_times))},
        "cases": {"both_right": both_right, "ens_only": ens_only,
                  "gpt_only": gpt_only, "both_wrong": both_wrong},
        "per_class": {cls: {
            "ensemble": sum(1 for i, s in enumerate(samples)
                            if s[1] == cls and ens_preds[i] == cls),
            "gpt": sum(1 for i, s in enumerate(samples)
                       if s[1] == cls and gpt_preds[i] == cls),
            "total": sum(1 for s in samples if s[1] == cls),
        } for cls in CLASSES},
        "results": [{
            "path": Path(s[0]).name,
            "gt": s[1],
            "ensemble": e,
            "gpt": g,
            "gpt_raw": r,
        } for s, e, g, r in zip(samples, ens_preds, gpt_preds, gpt_raws)],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
