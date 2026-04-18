#!/usr/bin/env python
"""E2E pipeline test — ViT 단일 + 앙상블 순차 실행.

val 샘플 4장 (class 1장씩) → predict → OpenAI GPT-5.1 스토리 → JSON 기록.

환경변수 OPENAI_API_KEY 필수. 코드/디스크에 key 저장 금지.
"""
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

PROJECT = Path("/workspace/user4/emotion-project")
sys.path.insert(0, str(PROJECT))

from predict import CLASSES, load_model, predict_probs  # noqa: E402
import numpy as np  # noqa: E402

OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or ""
assert OPENAI_KEY.startswith("sk-"), "OPENAI_API_KEY env 변수 필요"

SYSTEM_PROMPT = (
    "너는 자폐 스펙트럼 아동을 돕는 감정 학습 챗봇이야. "
    "주어진 감정(anger/happy/panic/sadness)에 맞는 "
    "짧고 구체적인 2문장짜리 일상 상황 이야기를 한국어로 만들어. "
    "가상 인물 이름 지어내지 말고, '너'로 지칭. 이모지 금지."
)


def call_openai(emotion: str, confidence: float) -> dict:
    """GPT-5.1 에 스토리 요청."""
    user_msg = (
        f"감정: {emotion}\n"
        f"모델 확신도: {confidence:.2f}\n\n"
        f"이 감정에 맞는 2문장 상황 이야기를 만들어줘."
    )
    payload = {
        "model": "gpt-5.1",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_completion_tokens": 200,
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
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    story = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {})
    return {"story": story, "llm_sec": round(dt, 2), "usage": usage}


def pick_samples(seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    for cls in CLASSES:
        imgs = sorted((PROJECT / "data_rot/img/val" / cls).glob("*.jpg"))
        out.append((str(rng.choice(imgs)), cls))
    return out


def run_one_model(model_name: str, model_path: str, samples: list,
                  warmup: int = 2) -> dict:
    print(f"\n{'='*72}")
    print(f"[MODEL] {model_name}  ({Path(model_path).name})")
    print(f"{'='*72}")

    # === COLD: load ===
    t0 = time.perf_counter()
    m = load_model(model_path, tta=False, auto_face_crop=True)
    load_sec = round(time.perf_counter() - t0, 2)
    print(f"  load {load_sec}s  type={m['type']}")

    # === WARMUP (XLA compile 비용 흡수, MLPerf 표준) ===
    warmup_sec = 0.0
    t0 = time.perf_counter()
    for i in range(warmup):
        _ = predict_probs(m, samples[i % len(samples)][0])
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    warmup_sec = round(time.perf_counter() - t0, 2)
    cold_total_sec = round(load_sec + warmup_sec, 2)
    print(f"  warmup {warmup}회 {warmup_sec}s → COLD total {cold_total_sec}s")

    # === WARM: 4 샘플 측정 ===
    results = []
    for path, gt in samples:
        t0 = time.perf_counter()
        probs = predict_probs(m, path)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        model_sec = round(time.perf_counter() - t0, 2)
        idx = int(np.argmax(probs))
        pred = CLASSES[idx]
        conf = float(probs[idx])
        llm = call_openai(pred, conf)
        total_sec = round(model_sec + llm["llm_sec"], 2)
        row = {
            "image": Path(path).name,
            "gt": gt,
            "pred": pred,
            "correct": pred == gt,
            "confidence": round(conf, 3),
            "probs": {c: round(float(probs[i]), 3) for i, c in enumerate(CLASSES)},
            "model_sec": model_sec,
            "llm_sec": llm["llm_sec"],
            "total_sec": total_sec,
            "story": llm["story"],
            "tokens": llm["usage"].get("total_tokens"),
        }
        results.append(row)
        mark = "✓" if pred == gt else "✗"
        print(f"  {mark} {gt:8s} → {pred:8s} ({conf:.3f})  model={model_sec}s  llm={llm['llm_sec']}s  total={total_sec}s")
        print(f"    story: {llm['story'][:70]}...")

    lat_model = [r["model_sec"] for r in results]
    lat_llm = [r["llm_sec"] for r in results]
    lat_total = [r["total_sec"] for r in results]
    acc = sum(r["correct"] for r in results) / len(results)
    confs = [r["confidence"] for r in results]
    sorted_model = sorted(lat_model)
    p95_idx = max(0, int(round(0.95 * (len(sorted_model) - 1))))
    return {
        "model_name": model_name,
        "model_path": model_path,
        "load_sec": load_sec,
        "warmup_sec": warmup_sec,
        "cold_total_sec": cold_total_sec,
        "accuracy": round(acc, 3),
        "mean_model_sec": round(sum(lat_model) / len(lat_model), 2),
        "median_model_sec": round(sorted_model[len(sorted_model) // 2], 2),
        "p95_model_sec": round(sorted_model[p95_idx], 2),
        "max_model_sec": round(max(lat_model), 2),
        "mean_llm_sec": round(sum(lat_llm) / len(lat_llm), 2),
        "mean_total_sec": round(sum(lat_total) / len(lat_total), 2),
        "mean_confidence": round(sum(confs) / len(confs), 3),
        "results": results,
    }


def main():
    samples = pick_samples(seed=42)
    print("samples:")
    for p, g in samples:
        print(f"  {g}: {Path(p).name}")

    vit = run_one_model(
        "ViT-B/16 single",
        str(PROJECT / "models/exp05_vit_b16_two_stage.pt"),
        samples,
    )
    ens = run_one_model(
        "Ensemble (4-model)",
        str(PROJECT / "models/ensemble_with_kd.json"),
        samples,
    )

    # 비교 분석
    print(f"\n{'='*72}")
    print("COMPARISON")
    print(f"{'='*72}")
    print(f"{'':35s} {'ViT':>12s} {'Ensemble':>12s}")
    print(f"{'-- COLD (1회 only) --':35s}")
    print(f"{'  load (s)':35s} {vit['load_sec']:>12.2f} {ens['load_sec']:>12.2f}")
    print(f"{'  warmup (s)':35s} {vit['warmup_sec']:>12.2f} {ens['warmup_sec']:>12.2f}")
    print(f"{'  cold total (load+warmup)':35s} {vit['cold_total_sec']:>12.2f} {ens['cold_total_sec']:>12.2f}")
    print(f"{'-- WARM (서비스 기준) --':35s}")
    print(f"{'  accuracy':35s} {vit['accuracy']:>12.3f} {ens['accuracy']:>12.3f}")
    print(f"{'  mean confidence':35s} {vit['mean_confidence']:>12.3f} {ens['mean_confidence']:>12.3f}")
    print(f"{'  model mean (s)':35s} {vit['mean_model_sec']:>12.2f} {ens['mean_model_sec']:>12.2f}")
    print(f"{'  model median p50 (s)':35s} {vit['median_model_sec']:>12.2f} {ens['median_model_sec']:>12.2f}")
    print(f"{'  model p95 (s)':35s} {vit['p95_model_sec']:>12.2f} {ens['p95_model_sec']:>12.2f}")
    print(f"{'  model max (s)':35s} {vit['max_model_sec']:>12.2f} {ens['max_model_sec']:>12.2f}")
    print(f"{'  LLM mean (s)':35s} {vit['mean_llm_sec']:>12.2f} {ens['mean_llm_sec']:>12.2f}")
    print(f"{'  e2e total mean (s)':35s} {vit['mean_total_sec']:>12.2f} {ens['mean_total_sec']:>12.2f}")

    # agreement
    agree = sum(1 for v, e in zip(vit["results"], ens["results"]) if v["pred"] == e["pred"])
    print(f"\npredict agreement: {agree}/{len(samples)}")

    # JSON 저장 (key 는 저장 안 됨, 환경변수만 썼으니)
    out = {
        "samples": [{"image": p, "gt": g} for p, g in samples],
        "vit": vit,
        "ensemble": ens,
        "prediction_agreement": f"{agree}/{len(samples)}",
    }
    out_path = PROJECT / "results/e2e_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
