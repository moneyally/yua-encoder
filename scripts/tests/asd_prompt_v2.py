#!/usr/bin/env python
"""ASD 감정 학습 파이프라인 — 프로덕션급 프롬프트 v2.

시나리오: 챗봇 상황 제시 → 아이가 표정 지음 → 카메라 캡처 → MTCNN 크롭 →
         앙상블 감정 분류 → 분기 결정 → GPT 프롬프트 구성 → LLM 응답 → 검증.

ABA / Discrete Trial Teaching 방법론 기반 system prompt.
4 분기 (ACCEPT / PARTIAL / MISMATCH / UNCERTAIN) 에 stabilize (panic 고확신도)
추가. Few-shot 3개 + JSON output_format 강제 + 후처리 검증 (이모지·가상이름·길이).

사용:
  python scripts/tests/asd_prompt_v2.py --gpt-model gpt-5.4
  python scripts/tests/asd_prompt_v2.py --gpt-model gpt-5.1 \\
      --output results/asd_prompt_v2_gpt51.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT = Path("/workspace/user4/emotion-project")
sys.path.insert(0, str(PROJECT))

from predict import CLASSES, detect_all_faces, load_model, predict_probs  # noqa: E402

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# -- label 한국어 매핑 (ASD 어휘 기준) ---------------------------------------
EN_TO_KO = {"anger": "화남", "happy": "기쁨", "panic": "놀람", "sadness": "슬픔"}
KO_TO_EN = {v: k for k, v in EN_TO_KO.items()}
VALID_KO = set(EN_TO_KO.values())

# -- 시스템 프롬프트 --------------------------------------------------------
SYSTEM_PROMPT = """\
# ROLE
너는 자폐 스펙트럼 아동 (6~10세) 의 감정 표현 학습을 돕는 한국어 챗봇 "마음이" 이다.
ABA (Applied Behavior Analysis) 와 Discrete Trial Teaching (DTT) 방법론을 따른다.

# PRIMARY GOAL
아이가 "상황 → 해당 감정 → 얼굴 표정" 의 연결을 반복 학습으로 내재화하도록
긍정 강화 중심으로 지도한다.

# LANGUAGE RULES
- 초등 1~2학년 수준 한국어만 사용
- 한 문장 최대 15자, 어려운 한자어 금지
- 외래어 최소 (happy → "좋은 기분", panic → "무서운 기분")
- 주어는 항상 "너"
- 이모지·특수기호 사용 금지

# ABA FEEDBACK RULES
- 정답: 즉각 구체적 칭찬 + 왜 맞았는지 쉬운 설명
- 오답: 비난·부정 표현 금지. "다른 감정을 표현했네" 중립 표현
- 재시도 유도는 격려 톤. "틀렸어" 대신 "다시 해볼까"
- 가상 인물 이름·상상 친구 절대 생성 금지

# SAFETY GUARD
- 자해·폭력·성적 내용 일체 금지
- 아이 표정이 놀람으로 확신도 > 0.9 면 "괜찮아 천천히 숨쉬자" 안정화
- 의료·진단 언어 금지

# OUTPUT FORMAT (JSON, 반드시 준수)
{
  "speech": "아이에게 소리내어 읽어줄 2문장 이내 피드백",
  "emotion_label_ko": "화남|기쁨|놀람|슬픔 중 하나",
  "is_correct": true|false,
  "next_action": "advance|retry|stabilize",
  "praise_intensity": "high|medium|low",
  "internal_note": "부모·교사 로그용 한 줄 (아이한테 안 보임)"
}

출력은 JSON 오브젝트 하나만. 설명·주석 금지.

# EXAMPLES

## Example 1 (ACCEPT, 화남)
INPUT: 상황 "친구가 네가 아끼는 장난감을 뺏어갔어" / 기대 화남 / 표정 화남 0.93 / ACCEPT
OUTPUT: {"speech":"잘했어. 눈썹이 아래로 내려갔네. 소중한 걸 뺏기면 화가 나는 게 맞아.","emotion_label_ko":"화남","is_correct":true,"next_action":"advance","praise_intensity":"high","internal_note":"표정 명확, 정답 매칭"}

## Example 2 (MISMATCH, 기대=기쁨 / 표정=놀람)
INPUT: 상황 "선생님이 너에게 칭찬 스티커를 주셨어" / 기대 기쁨 / 표정 놀람 0.88 / MISMATCH
OUTPUT: {"speech":"놀란 표정을 지었구나. 기쁠 땐 입꼬리가 올라가.","emotion_label_ko":"놀람","is_correct":false,"next_action":"retry","praise_intensity":"low","internal_note":"놀람/기쁨 혼동, 입 근육 단서 힌트 제공"}

## Example 3 (UNCERTAIN, 확신도 낮음)
INPUT: 상황 "친한 친구가 멀리 이사 간대" / 기대 슬픔 / 표정 슬픔 0.42 / UNCERTAIN
OUTPUT: {"speech":"얼굴이 잘 안 보여. 카메라를 똑바로 보고 다시 해볼까.","emotion_label_ko":"슬픔","is_correct":false,"next_action":"retry","praise_intensity":"low","internal_note":"confidence 0.42, 자세/조명 이슈 추정"}
"""

# -- 시나리오 Bank ----------------------------------------------------------
SCENARIOS: list[dict] = [
    {"expected_ko": "화남",   "text": "누가 너를 때렸어. 기분이 어때?"},
    {"expected_ko": "화남",   "text": "동생이 네 장난감을 허락 없이 가져갔어. 기분이 어때?"},
    {"expected_ko": "기쁨",   "text": "선생님이 너에게 반짝이 스티커를 주셨어. 기분이 어때?"},
    {"expected_ko": "기쁨",   "text": "엄마가 네가 좋아하는 피자를 준비했어. 기분이 어때?"},
    {"expected_ko": "놀람",   "text": "갑자기 큰 소리가 나고 불이 꺼졌어. 기분이 어때?"},
    {"expected_ko": "놀람",   "text": "친구가 뒤에서 깜짝 놀래켰어. 기분이 어때?"},
    {"expected_ko": "슬픔",   "text": "친한 친구가 먼 동네로 이사 간대. 기분이 어때?"},
    {"expected_ko": "슬픔",   "text": "아끼던 반려동물이 아파서 병원에 갔어. 기분이 어때?"},
]

# -- 분기 지침 --------------------------------------------------------------
BRANCH_GUIDE = {
    "ACCEPT":
        "- 구체적 칭찬 1문장 (아이 행동 단서를 콕 짚어 인정)\n"
        "- 왜 그 감정이 맞는지 쉬운 신체 단서 설명 1문장\n"
        "- is_correct=true, next_action=advance, praise_intensity=high",
    "PARTIAL":
        "- 맞췄다는 확인 1문장\n"
        "- 표정을 조금 더 크게 지어보자고 부드럽게 유도 1문장\n"
        "- is_correct=true, next_action=retry, praise_intensity=medium",
    "MISMATCH":
        "- '다른 감정을 표현했구나' 중립 확인 1문장\n"
        "- 기대 감정의 신체 단서 설명 1문장 (예: 화남 → 눈썹이 아래로)\n"
        "- is_correct=false, next_action=retry, praise_intensity=low",
    "UNCERTAIN":
        "- 카메라/자세 때문에 표정이 잘 안 보인다는 걸 아이 탓 안 하게 전달\n"
        "- 다시 한 번 해보자고 격려\n"
        "- is_correct=false, next_action=retry, praise_intensity=low",
    "STABILIZE":
        "- 놀람 강도가 매우 높으니 '괜찮아 천천히 숨쉬자' 진정 유도\n"
        "- 깊게 숨쉬는 쉬운 가이드 1문장\n"
        "- is_correct=true, next_action=stabilize, praise_intensity=medium",
}


def decide_branch(expected_ko: str, pred_ko: str, conf: float) -> str:
    """예측 결과로 분기 결정."""
    if pred_ko == "놀람" and conf >= 0.90:
        return "STABILIZE"
    if pred_ko == expected_ko and conf >= 0.85:
        return "ACCEPT"
    if pred_ko == expected_ko and conf >= 0.60:
        return "PARTIAL"
    if conf < 0.60:
        return "UNCERTAIN"
    return "MISMATCH"


def build_user_prompt(scenario: str, expected_ko: str, pred_ko: str, conf: float,
                      branch: str, face_bbox: str = "전체 이미지") -> str:
    return (
        "## 현재 시도\n"
        f"- 제시 상황: \"{scenario}\"\n"
        f"- 기대 감정: {expected_ko}\n"
        f"- 아이 표정 분석 결과: {pred_ko} (확신도 {conf:.2f})\n"
        f"- 얼굴 검출 bbox: {face_bbox}\n"
        f"- 판정 분기: {branch}\n"
        "\n"
        "## 분기별 지침\n"
        f"{BRANCH_GUIDE[branch]}\n"
        "\n"
        "## 요청\n"
        "위 규칙·분기 지침에 맞춰 JSON 한 오브젝트만 생성하라."
    )


# -- LLM 호출 ---------------------------------------------------------------
def call_llm(user_prompt: str, model: str, seed: int = 42,
             timeout: int = 60) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 300,
        "seed": seed,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {OPENAI_KEY}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    raw = (body["choices"][0]["message"]["content"] or "").strip()
    usage = body.get("usage", {})
    return {"raw": raw, "dt_sec": round(dt, 2),
            "tokens": usage.get("total_tokens")}


# -- 검증 -------------------------------------------------------------------
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]")
FAKE_NAME_RE = re.compile(r"(민수|영희|철수|지민|영수|수진|하나|민지)")
SENT_SPLIT = re.compile(r"[.!?]")


def validate_output(raw: str) -> tuple[dict | None, list[str]]:
    """JSON parse + 규칙 검증. (payload, violations)."""
    violations: list[str] = []
    try:
        out = json.loads(raw)
    except Exception as e:
        return None, [f"json_parse_fail: {e!r}"]

    for k in ("speech", "emotion_label_ko", "is_correct",
              "next_action", "praise_intensity", "internal_note"):
        if k not in out:
            violations.append(f"missing_field: {k}")

    if "speech" in out:
        spk = out["speech"]
        if not isinstance(spk, str) or not spk.strip():
            violations.append("speech_empty")
        else:
            if EMOJI_RE.search(spk):
                violations.append("emoji_present")
            if FAKE_NAME_RE.search(spk):
                violations.append("fake_name")
            sents = [s for s in SENT_SPLIT.split(spk) if s.strip()]
            if len(sents) > 2:
                violations.append(f"too_many_sentences({len(sents)})")
            if len(spk) > 80:
                violations.append(f"speech_too_long({len(spk)})")

    if out.get("emotion_label_ko") not in VALID_KO:
        violations.append(f"invalid_emotion_ko: {out.get('emotion_label_ko')!r}")

    if out.get("next_action") not in ("advance", "retry", "stabilize"):
        violations.append(f"invalid_next_action: {out.get('next_action')!r}")

    if out.get("praise_intensity") not in ("high", "medium", "low"):
        violations.append(f"invalid_praise: {out.get('praise_intensity')!r}")

    if not isinstance(out.get("is_correct"), bool):
        violations.append("is_correct_not_bool")

    return out, violations


# -- val 이미지 픽 ---------------------------------------------------------
def pick_val_image(expected_en: str, rng: random.Random) -> str:
    cls_dir = PROJECT / "data_rot/img/val" / expected_en
    imgs = sorted(cls_dir.glob("*.jpg"))
    return str(rng.choice(imgs))


# -- 메인 루프 --------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-model", default="gpt-5.4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(PROJECT / "results/asd_prompt_v2.json"))
    ap.add_argument("--ensemble", default=str(PROJECT / "models/ensemble_with_kd.json"))
    args = ap.parse_args()

    if not OPENAI_KEY.startswith("sk-"):
        print("ERROR: OPENAI_API_KEY env 변수 필요", file=sys.stderr)
        return 1

    print(f"[setup] gpt={args.gpt_model}  seed={args.seed}  "
          f"scenarios={len(SCENARIOS)}", file=sys.stderr)

    # 모델 로드 + warmup
    print("[load] ensemble ...", file=sys.stderr)
    t0 = time.perf_counter()
    model = load_model(args.ensemble, tta=False, auto_face_crop=True)
    print(f"  load {time.perf_counter() - t0:.1f}s  type={model['type']}",
          file=sys.stderr)

    rng = random.Random(args.seed)
    # warmup 2
    tmp_path = pick_val_image("anger", random.Random(0))
    print("[warmup] 2회 (XLA compile)", file=sys.stderr)
    for _ in range(2):
        _ = predict_probs(model, tmp_path)

    # 실행
    print(f"\n{'='*92}", file=sys.stderr)
    print(f"{'#':>2s} {'expected':8s} {'pred':6s} {'conf':>6s} {'branch':>10s} "
          f"{'valid':>6s} {'sec':>5s}  output", file=sys.stderr)
    print("=" * 92, file=sys.stderr)

    results = []
    for i, sc in enumerate(SCENARIOS, 1):
        expected_ko = sc["expected_ko"]
        expected_en = KO_TO_EN[expected_ko]
        img_path = pick_val_image(expected_en, rng)

        # 앙상블 예측
        probs = predict_probs(model, img_path)
        pred_idx = int(np.argmax(probs))
        pred_en = CLASSES[pred_idx]
        pred_ko = EN_TO_KO[pred_en]
        conf = float(probs.max())

        # 얼굴 bbox (optional, 로그용)
        try:
            from PIL import Image as _I
            pil = _I.open(img_path).convert("RGB")
            faces = detect_all_faces(pil, margin=0.1, min_face_size=20,
                                     min_conf=0.9, device="cuda")
            bbox = str(faces[0]["bbox"]) if faces else "전체 이미지"
        except Exception:
            bbox = "전체 이미지"

        branch = decide_branch(expected_ko, pred_ko, conf)
        user_prompt = build_user_prompt(sc["text"], expected_ko, pred_ko,
                                         conf, branch, bbox)

        try:
            llm = call_llm(user_prompt, args.gpt_model, seed=args.seed)
        except Exception as e:
            print(f"  [llm_err] {e!r}", file=sys.stderr)
            results.append({
                "idx": i, "scenario": sc["text"],
                "expected_ko": expected_ko, "pred_ko": pred_ko,
                "conf": round(conf, 3), "branch": branch,
                "file": Path(img_path).name,
                "llm_error": str(e)[:200],
            })
            continue

        payload, violations = validate_output(llm["raw"])
        valid = not violations

        mark_v = "OK" if valid else "X "
        preview = payload["speech"] if payload else llm["raw"][:40]
        print(f"{i:>2d} {expected_ko:8s} {pred_ko:6s} {conf:>6.3f} "
              f"{branch:>10s} {mark_v:>6s} {llm['dt_sec']:>4.1f}s  "
              f"{preview[:52]}", file=sys.stderr)

        results.append({
            "idx": i, "scenario": sc["text"],
            "expected_ko": expected_ko, "pred_ko": pred_ko,
            "conf": round(conf, 3), "branch": branch,
            "file": Path(img_path).name,
            "face_bbox": bbox,
            "llm_dt_sec": llm["dt_sec"],
            "llm_tokens": llm["tokens"],
            "llm_raw": llm["raw"],
            "payload": payload,
            "violations": violations,
            "valid": valid,
        })

    # 집계
    print(f"\n{'='*92}", file=sys.stderr)
    n_valid = sum(1 for r in results if r.get("valid"))
    n_total = len(results)
    n_branches: dict[str, int] = {}
    for r in results:
        n_branches[r["branch"]] = n_branches.get(r["branch"], 0) + 1
    avg_dt = np.mean([r["llm_dt_sec"] for r in results if "llm_dt_sec" in r])

    print(f"유효 JSON + 규칙 통과: {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)")
    print(f"분기 분포: {n_branches}")
    print(f"LLM 평균 latency: {avg_dt:.1f}s")

    # violation 유형별
    v_count: dict[str, int] = {}
    for r in results:
        for v in r.get("violations", []):
            key = v.split(":")[0]
            v_count[key] = v_count.get(key, 0) + 1
    if v_count:
        print(f"\n규칙 위반 유형:")
        for k, c in sorted(v_count.items(), key=lambda x: -x[1]):
            print(f"  {k}: {c}")

    # 상세 sample
    print(f"\n샘플 응답 3건:")
    for r in results[:3]:
        if not r.get("payload"): continue
        print(f"  [{r['idx']}] {r['expected_ko']} / 분기 {r['branch']}")
        print(f"      시나리오: {r['scenario']}")
        print(f"      모델 예측: {r['pred_ko']} (conf {r['conf']})")
        print(f"      챗봇: {r['payload']['speech']}")
        print(f"      next_action: {r['payload']['next_action']}  "
              f"praise: {r['payload']['praise_intensity']}")
        print()

    # 저장
    out = {
        "gpt_model": args.gpt_model,
        "seed": args.seed,
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_rate": round(n_valid / n_total, 3),
        "branch_distribution": n_branches,
        "violation_types": v_count,
        "mean_llm_latency_sec": round(float(avg_dt), 2),
        "scenarios": SCENARIOS,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
