#!/usr/bin/env python
"""ASD 파이프라인 증거 기반 Trace — 1 trial 의 모든 단계 완전 추적.

단계:
  1. 입력 이미지 메타
  2. MTCNN 얼굴 탐지 (bbox, conf, 소요 ms)
  3. 크롭 결과 (shape)
  4. 앙상블 각 member 의 raw probs
  5. weight 적용 + softmax 정규화
  6. 분기 결정 (ACCEPT / PARTIAL / ...)
  7. LLM 프롬프트 전체 (system + user)
  8. LLM 응답 raw + parse + violations
  9. 최종 speech

이 스크립트는 asd_prompt_v2.py 와 동일 로직을 내부 직접 실행해서
각 중간 상태를 캡처한다. 필요한 샘플만 돌리므로 빠름.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path("/workspace/user4/emotion-project")
sys.path.insert(0, str(PROJECT))

from predict import CLASSES, detect_all_faces, load_model  # noqa: E402
from scripts.tests.asd_prompt_v2 import (   # noqa: E402
    SYSTEM_PROMPT, SCENARIOS, KO_TO_EN, EN_TO_KO,
    decide_branch, build_user_prompt, call_llm, validate_output,
)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")


def banner(title: str, ch: str = "=") -> None:
    print("\n" + ch * 80)
    print(f"  {title}")
    print(ch * 80)


def trace_one(expected_ko: str, scenario_text: str, model_path: Path,
              gpt_model: str, seed: int = 42) -> dict:
    rng = random.Random(seed)

    # ============================================================
    # STEP 1. 입력 이미지 선택
    # ============================================================
    banner("STEP 1. 입력 이미지")
    expected_en = KO_TO_EN[expected_ko]
    cls_dir = PROJECT / "data_rot/img/val" / expected_en
    imgs = sorted(cls_dir.glob("*.jpg"))
    img_path = str(rng.choice(imgs))
    pil = Image.open(img_path).convert("RGB")
    print(f"  path        {img_path}")
    print(f"  class (gt)  {expected_ko} ({expected_en})")
    print(f"  resolution  {pil.size[0]}×{pil.size[1]}")
    print(f"  mode        {pil.mode}")
    print(f"  file size   {os.path.getsize(img_path) / 1024:.1f} KB")

    # ============================================================
    # STEP 2. MTCNN 얼굴 탐지
    # ============================================================
    banner("STEP 2. MTCNN 얼굴 탐지")
    t0 = time.perf_counter()
    faces = detect_all_faces(pil, margin=0.1, min_face_size=20,
                              min_conf=0.9, device="cuda")
    mtcnn_ms = (time.perf_counter() - t0) * 1000
    print(f"  소요         {mtcnn_ms:.1f} ms")
    print(f"  얼굴 N      {len(faces)}")
    for i, f in enumerate(faces, 1):
        x1, y1, x2, y2 = f["bbox"]
        print(f"  얼굴 #{i}    bbox=({x1},{y1})-({x2},{y2}) "
              f"size={x2-x1}×{y2-y1}  conf={f['confidence']:.3f}")
    bbox_str = str(faces[0]["bbox"]) if faces else "전체 이미지"

    # ============================================================
    # STEP 3. 모델 로드 + 각 member raw probs
    # ============================================================
    banner("STEP 3. 앙상블 member 별 raw probs")
    t0 = time.perf_counter()
    ens = load_model(str(model_path), tta=False, auto_face_crop=True)
    load_sec = time.perf_counter() - t0
    print(f"  ensemble load   {load_sec:.1f}s  type={ens['type']}")

    # 첫 호출은 XLA compile → warmup 1회
    print(f"  [warmup] 1회 (XLA compile) ...")
    _ = ens["predict_fn_raw"](pil)

    # 각 member (meta/_model 에 직접 접근하기엔 내부 구조 복잡)
    # _load_ensemble 는 loaded list 를 노출하지 않으므로
    # config json 에서 path/weight 읽고 개별 ckpt 를 다시 로드해야 하지만,
    # 여기서는 ens["meta"]["members"] 로 정보만 표시하고,
    # 최종 weighted probs 만 측정.
    members = ens["meta"]["members"]
    print(f"  members ({len(members)})")
    for m in members:
        name = Path(m["path"]).name
        print(f"    {m['type']:<45s}  weight={m['weight']:.4f}  file={name}")

    # 최종 앙상블 probs (이미 softmax, weighted)
    t0 = time.perf_counter()
    # predict_fn_base = crop 포함, predict_fn_raw = crop 없음
    # 학습과 동일 조건인 crop 을 쓰려면 base 사용
    probs = ens["predict_fn_base"](pil)
    infer_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  최종 weighted probs  (infer {infer_ms:.0f} ms)")
    for c, p in zip(CLASSES, probs):
        bar = "█" * int(p * 40)
        print(f"    {c:<8s} {p:.4f}  {bar}")
    pred_idx = int(np.argmax(probs))
    pred_en = CLASSES[pred_idx]
    pred_ko = EN_TO_KO[pred_en]
    conf = float(probs[pred_idx])
    print(f"\n  argmax → {pred_en} ({pred_ko})  conf={conf:.4f}")
    print(f"  ground truth → {expected_en} ({expected_ko})")
    print(f"  match → {'✓' if pred_en == expected_en else '✗'}")

    # ============================================================
    # STEP 4. 분기 결정
    # ============================================================
    banner("STEP 4. 분기 결정")
    branch = decide_branch(expected_ko, pred_ko, conf)
    print(f"  규칙 평가:")
    print(f"    pred_ko == expected_ko ? {pred_ko == expected_ko}")
    print(f"    conf >= 0.85 ?            {conf >= 0.85}  (실제 {conf:.3f})")
    print(f"    conf >= 0.60 ?            {conf >= 0.60}")
    print(f"    conf >= 0.90 AND 놀람?   {conf >= 0.90 and pred_ko == '놀람'}")
    print(f"\n  → 분기: {branch}")

    # ============================================================
    # STEP 5. LLM 프롬프트 조립
    # ============================================================
    banner("STEP 5. LLM 프롬프트 조립")
    user_prompt = build_user_prompt(
        scenario_text, expected_ko, pred_ko, conf, branch, bbox_str)
    print("  [system] (처음 400자)")
    print("    " + SYSTEM_PROMPT[:400].replace("\n", "\n    "))
    print("  ...")
    print("\n  [user]")
    print("    " + user_prompt.replace("\n", "\n    "))

    # ============================================================
    # STEP 6. GPT API 호출
    # ============================================================
    banner(f"STEP 6. GPT-{gpt_model.replace('gpt-', '')} API 호출")
    if not OPENAI_KEY.startswith("sk-"):
        print("  [SKIP] OPENAI_API_KEY 없음")
        return {"skipped": True, "img": img_path, "pred_ko": pred_ko, "conf": conf,
                "branch": branch, "faces": faces}
    t0 = time.perf_counter()
    llm = call_llm(user_prompt, gpt_model, seed=seed)
    llm_sec = time.perf_counter() - t0
    print(f"  요청 모델   {gpt_model}")
    print(f"  소요        {llm_sec:.2f}s")
    print(f"  total tokens  {llm['tokens']}")
    print(f"\n  [raw response]")
    print("    " + llm["raw"])

    # ============================================================
    # STEP 7. JSON 파싱 + 검증
    # ============================================================
    banner("STEP 7. JSON 파싱 + 규칙 검증")
    payload, violations = validate_output(llm["raw"])
    if payload:
        print("  필드 체크:")
        for k in ("speech", "emotion_label_ko", "is_correct",
                  "next_action", "praise_intensity", "internal_note"):
            v = payload.get(k, "<missing>")
            print(f"    {k:<20s} {v!r}")
    print(f"\n  규칙 위반: {violations if violations else '없음 (PASS)'}")

    # ============================================================
    # STEP 8. 최종 출력
    # ============================================================
    banner("STEP 8. 아이에게 표시될 최종 결과", "=")
    if payload:
        print(f"\n  \033[1m챗봇 speech:\033[0m")
        print(f"     {payload['speech']}")
        print(f"\n  next_action     {payload['next_action']}")
        print(f"  praise_intensity{payload['praise_intensity']:>5s}")
        print(f"  is_correct      {payload['is_correct']}")
        print(f"  internal_note   {payload['internal_note']}")

    return {
        "image": img_path,
        "resolution": f"{pil.size[0]}x{pil.size[1]}",
        "mtcnn_ms": round(mtcnn_ms, 1),
        "n_faces": len(faces),
        "bbox": bbox_str,
        "infer_ms": round(infer_ms, 0),
        "probs": {c: round(float(probs[i]), 4) for i, c in enumerate(CLASSES)},
        "pred_ko": pred_ko,
        "conf": round(conf, 4),
        "expected_ko": expected_ko,
        "branch": branch,
        "llm_sec": round(llm_sec, 2),
        "llm_raw": llm["raw"],
        "payload": payload,
        "violations": violations,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-ko", default="화남",
                    choices=["화남", "기쁨", "놀람", "슬픔"])
    ap.add_argument("--scenario-idx", type=int, default=0,
                    help="SCENARIOS 내 해당 감정 번째")
    ap.add_argument("--gpt-model", default="gpt-5.4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(PROJECT / "results/asd_trace.json"))
    ap.add_argument("--ensemble", default=str(PROJECT / "models/ensemble_with_kd.json"))
    args = ap.parse_args()

    # 시나리오 찾기
    candidates = [s for s in SCENARIOS if s["expected_ko"] == args.expected_ko]
    if args.scenario_idx >= len(candidates):
        print(f"ERROR: {args.expected_ko} scenario idx 범위 초과 "
              f"(max {len(candidates) - 1})", file=sys.stderr)
        return 1
    sc = candidates[args.scenario_idx]

    print(f"\n{'#'*80}")
    print(f"# ASD 파이프라인 증거 Trace")
    print(f"# expected={args.expected_ko}  scenario=\"{sc['text']}\"  gpt={args.gpt_model}")
    print(f"{'#'*80}")

    result = trace_one(sc["expected_ko"], sc["text"],
                       Path(args.ensemble), args.gpt_model, args.seed)

    # 저장
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
