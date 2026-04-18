#!/usr/bin/env python
"""B. 자폐 아동 감정 분류 파이프라인 실측.

CLAUDE.md #20 self-improving pipeline 시뮬레이션:
  1) 챗봇이 "슬픈 상황" 제시 → expected_emotion = sadness
  2) 아동이 표정 짓고 사진 촬영 → 앙상블로 예측
  3) 분기 필터:
       pred == expected AND conf ≥ 0.85   → ACCEPT (rehearsal buffer)
       conf < 0.6                         → RETRY
       else                               → DROP (self-poisoning 방지)
  4) 샘플 3개는 실제 GPT-5.1 에 자폐 8세 아동용 분기 응답 생성

val 20장 (class 5×4) 을 "아동 표정" 으로 가정하고 시뮬레이션.
"""
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
LLM_AVAILABLE = OPENAI_KEY.startswith("sk-")

# 분기 임계치 (CLAUDE.md #20 명세)
ACCEPT_CONF = 0.85
RETRY_CONF = 0.60

# 챗봇이 제시하는 상황 매핑 (expected emotion)
SCENARIOS = {
    "anger": "친구가 네가 아끼던 장난감을 망가뜨렸어",
    "happy": "가족이 네가 좋아하는 선물을 준비했어",
    "panic": "갑자기 교실 불이 꺼지고 큰 소리가 났어",
    "sadness": "친한 친구가 다른 동네로 이사 간대",
}

SYSTEM_PROMPT = (
    "너는 자폐 스펙트럼 8세 아동의 감정 학습을 돕는 한국어 챗봇이야. "
    "매우 쉬운 단어만 써. 한 문장은 15자 이내. 이모지 금지. "
    "가상 인물 이름 지어내지 말고 '너'로 지칭. "
    "부정적 판단·비난 금지. 항상 따뜻하고 구체적인 톤."
)


def call_llm(scenario: str, expected: str, pred: str, conf: float, decision: str) -> dict:
    """분기별 응답 생성.
      ACCEPT → 칭찬 + 감정 인식 확인
      RETRY  → 한번 더 해보자 유도
      DROP   → 모델 불확실 알림 + 부드러운 재시도
    """
    if decision == "ACCEPT":
        user = (
            f"상황: {scenario}\n"
            f"기대 감정: {expected}\n"
            f"아동이 지은 표정: {pred} (확신도 {conf:.2f})\n\n"
            "아동이 정답 감정을 잘 표현했어. 2문장으로 칭찬 + "
            f"왜 {expected} 감정이 맞는지 쉽게 설명해."
        )
    elif decision == "RETRY":
        user = (
            f"상황: {scenario}\n"
            f"기대 감정: {expected}\n"
            f"모델이 읽은 표정: {pred} (확신도 {conf:.2f} — 낮음)\n\n"
            "표정이 잘 안 보여. 2문장으로 다시 시도하자고 부드럽게 말해."
        )
    else:  # DROP
        user = (
            f"상황: {scenario}\n"
            f"기대 감정: {expected}\n"
            f"모델이 읽은 표정: {pred} (다른 감정, 확신도 {conf:.2f})\n\n"
            "아동이 다른 감정을 표현한 것 같아. 2문장으로 "
            f"\"{expected}\" 감정이 어떤 건지 다시 알려주고 같이 해보자고 말해."
        )
    payload = {
        "model": "gpt-5.1",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": 200,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return {
        "reply": body["choices"][0]["message"]["content"].strip(),
        "llm_sec": round(time.perf_counter() - t0, 2),
    }


def pick_samples(n_per_class: int = 5, seed: int = 42):
    rng = random.Random(seed)
    out = []
    for cls in CLASSES:
        imgs = sorted((PROJECT / "data_rot/img/val" / cls).glob("*.jpg"))
        for p in rng.sample(imgs, n_per_class):
            out.append((str(p), cls))
    return out


def main():
    samples = pick_samples(5, seed=42)
    print(f"[setup] samples={len(samples)}  LLM={'ON' if LLM_AVAILABLE else 'OFF'}")

    print("\n[load] ensemble 모델...")
    t0 = time.perf_counter()
    model = load_model(str(PROJECT / "models/ensemble_with_kd.json"),
                       tta=False, auto_face_crop=True)
    load_sec = time.perf_counter() - t0
    print(f"  load {load_sec:.1f}s")

    # warmup 2회 (XLA compile 비용 제거)
    print("[warmup] 2회...")
    for i in range(2):
        _ = predict_probs(model, samples[i][0])

    # === 20장 시뮬레이션 ===
    print("\n" + "=" * 80)
    print("[B-1] ASD 아동 학습 시뮬레이션 (val 20장 = 아동 표정 20회 촬영 가정)")
    print("=" * 80)
    print(f"{'expected':9s} {'predicted':10s} {'conf':>6s} {'decision':>8s}  scenario")
    print("-" * 80)

    results = []
    decisions = {"ACCEPT": 0, "RETRY": 0, "DROP": 0}
    for path, gt in samples:
        probs = predict_probs(model, path)
        idx = int(np.argmax(probs))
        pred = CLASSES[idx]
        conf = float(probs[idx])
        # 분기 결정
        if pred == gt and conf >= ACCEPT_CONF:
            decision = "ACCEPT"
        elif conf < RETRY_CONF:
            decision = "RETRY"
        else:
            decision = "DROP"
        decisions[decision] += 1
        results.append({
            "file": Path(path).name[:30],
            "expected": gt,
            "pred": pred,
            "conf": round(conf, 3),
            "decision": decision,
            "scenario": SCENARIOS[gt],
            "probs": {c: round(float(probs[i]), 3) for i, c in enumerate(CLASSES)},
        })
        print(f"{gt:9s} {pred:10s} {conf:>6.3f} {decision:>8s}  {SCENARIOS[gt][:40]}")

    # === 요약 ===
    print("\n" + "-" * 80)
    print("분기 통계")
    print("-" * 80)
    total = len(samples)
    for d, n in decisions.items():
        pct = n / total * 100
        print(f"  {d:<8s} {n}/{total}  ({pct:.1f}%)")
    acc = sum(1 for r in results if r["pred"] == r["expected"]) / total
    mean_conf = np.mean([r["conf"] for r in results])
    accept_mean_conf = np.mean([r["conf"] for r in results if r["decision"] == "ACCEPT"]) if decisions["ACCEPT"] else 0.0
    print(f"  raw accuracy: {acc*100:.1f}%")
    print(f"  mean confidence: {mean_conf:.3f}")
    print(f"  ACCEPT 의 평균 confidence: {accept_mean_conf:.3f} (rehearsal buffer 품질)")

    # === 샘플 3건 LLM 응답 ===
    if LLM_AVAILABLE:
        print("\n" + "=" * 80)
        print("[B-2] GPT-5.1 분기별 챗봇 응답 (샘플 3건 — ACCEPT/RETRY/DROP 각 1)")
        print("=" * 80)
        sample_per_decision = {}
        for r in results:
            d = r["decision"]
            if d not in sample_per_decision:
                sample_per_decision[d] = r
            if len(sample_per_decision) == 3:
                break
        for d, r in sample_per_decision.items():
            print(f"\n[{d}]  expected={r['expected']}  pred={r['pred']}  conf={r['conf']}")
            print(f"  상황: {r['scenario']}")
            try:
                llm = call_llm(r["scenario"], r["expected"], r["pred"], r["conf"], d)
                print(f"  챗봇: {llm['reply']}")
                print(f"  (llm {llm['llm_sec']}s)")
                r["llm_reply"] = llm["reply"]
                r["llm_sec"] = llm["llm_sec"]
            except Exception as e:
                print(f"  [ERR] {e}")

    # === 저장 ===
    out = {
        "n_samples": total,
        "thresholds": {"accept_conf": ACCEPT_CONF, "retry_conf": RETRY_CONF},
        "decisions": decisions,
        "accuracy": round(acc, 3),
        "mean_conf": round(float(mean_conf), 3),
        "accept_mean_conf": round(float(accept_mean_conf), 3),
        "results": results,
    }
    out_path = PROJECT / "results/asd_pipeline_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
