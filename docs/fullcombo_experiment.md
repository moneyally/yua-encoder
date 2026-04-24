# Full Combo 실험 보고서

**목적**: 현재 앙상블 val_acc 0.8692 (bbox DE) 를 뚫기 위한 3-rater Soft Label + Mixup + SWA 조합 실험.

**가설**: hard label 기반 학습은 라벨러 합의 ceiling (전원 0.6108 / 다수결 0.9045) 에 묶임. 3-rater 분포 자체를 학습 신호로 사용하면 "의견이 갈리는 샘플" 의 정보를 추가로 활용할 수 있어 ceiling 권역을 찌를 수 있다.

**시작**: 2026-04-22

---

## 1. 배경 분석

### 1.1 현재 앙상블 구성 (E8 메인 제출)

| Member | 단독 val_acc | weight |
|---|---:|---:|
| E2 ResNet50 ft | 0.7758 | 0.041 |
| E4 EfficientNet-B0 ft | 0.7933 | 0.073 |
| E5 ViT-B/16 2-stage | 0.8383 | 0.523 |
| E9 SigLIP + KD (T=4, α=0.7, uf=4) | 0.8333 | 0.363 |
| **앙상블 (DE optimized)** | **0.8692** | — |

### 1.2 시도한 SOTA 전이 실패

외부 FER pretrained 모델을 val 에 적용한 결과:
- BEiT-L (FER2013+RAF-DB+AffectNet, HF) → **0.6892** (MTCNN crop)
- DDAMFN++ (RAF-DB 92% pretrained) → **0.6617** (MTCNN crop)

두 모델 모두 한국인 스튜디오 얼굴 도메인과 서양/in-the-wild 데이터 간 distribution gap 때문에 zero-shot 전이가 실패. 이 결과는 **"우리 데이터에 맞춘 학습이 유일한 경로"** 라는 반증 증거로 활용.

### 1.3 라벨러 합의 실측 (soft labels 생성 결과)

| split | n | 3/3 일치 | 2/3 | 1/3 | 0/3 fallback | argmax(soft)=GT |
|---|---:|---:|---:|---:|---:|---:|
| train | 5,994 | 5,394 (90.0%) | 442 | 124 | 34 | 89.7% |
| val | 1,200 | 1,064 (88.7%) | 90 | 30 | 16 | 88.9% |

실측 전원일치율은 README 에 기록된 0.6108 보다 높음. 차이: README 수치는 면밀한 검증 후 엄격 조건, soft label 생성은 우리 매핑 (분노/화남/격분 → anger 등) 확장 후 수치.

---

## 2. 설계

### 2.1 기법 3개 선택 근거

| 기법 | 논문 | 예상 기여 | 우리 상황 적용 이유 |
|---|---|---|---|
| **3-rater Soft Label** | LDL (TPAMI 2016), LDL-ALSG (CVPR 2020) | +0.8~1.5%p | ceiling 뚫는 유일한 정공법. annot_A/B/C 는 우리만 가진 자원 |
| **Mixup** (α=0.2) | ICLR 2018, MixAugment arxiv 2205.04442 | +0.3~0.8%p | 7,196장 소규모 데이터 overfit 방어. α=0.2 로 표정 보존 |
| **SWA** | UAI 2018, arxiv 1803.05407 | +0.3~0.8%p | flat minima 수렴. fine-tune 후반 평균으로 안정화 |

### 2.2 Loss 설계

```
Loss = α · KL(soft_3rater ∥ student) + (1 - α) · KL(hard_onehot ∥ student)
α = 0.5
```

- soft 와 hard 둘 다 probability distribution (sum=1). `-(target · log_softmax(logits)).sum()` 형태.
- hard one-hot 기준 KL = Cross-Entropy 와 수학적으로 동등 (QA Agent 검증됨).
- Mixup 적용 시 x, soft, hard 셋 모두 동일 lam 으로 혼합 → 일관성 유지.

### 2.3 학습 스케줄 (2-stage fine-tune)

```
Stage A (ep 1~5): backbone freeze, head only
  lr_head=1e-3, cosine warmup 2, Mixup OFF, SWA OFF

Stage B (ep 6~20, 총 15 epoch): backbone 전체 unfreeze
  lr_backbone=5e-5, lr_head=1e-4, cosine + warmup 2
  ep 6~11: Mixup α=0.2, prob=0.5
  ep 12~19: + SWA (AveragedModel 에 weight 누적)
  ep 20: 마지막 SWA update + update_bn + evaluate
```

### 2.4 재현성

- seed 42 고정 (torch / numpy / random / DataLoader worker init)
- bf16 mixed precision
- cudnn.benchmark=True (속도 우선, 재현 오차 ±0.3%p 예상 → 보고서에 명시)
- 동일 val 1,200 fixed split
- 단일 seed 학습이므로 seed 변동 효과는 ablation 에 포함하지 않음

### 2.5 QA Agent 검토 결과 (선행)

- Critical 버그: 0건
- Major 이슈: 5건 중 **Loss 수학 정합성 ✅ 확인**, 나머지는 2차 실행에서 개선
- 판정: **PASS (조건부)**

---

## 3. 실행 환경

| | |
|---|---|
| GPU | NVIDIA A40 48GB |
| PyTorch | 2.6.0+cu124 |
| timm | 1.0.26 |
| 총 예상 시간 | ViT 3~4h |
| 학습 시작 | 2026-04-22 |

---

## 4. 결과

### 4.1 학습 곡선 (요약)

Stage A (head only, 5 ep): val_acc 0.5725 → 0.7017  
Stage B (backbone unfreeze, 15 ep): val_acc 0.7217 → 0.8608 (B11), 최종 ep B15 = 0.8583.  
ep B7 부터 SWA 누적 시작. ep B11 (global ep16) 에서 best val_acc 달성.

csv: `logs/exp10_vit_fullcombo.csv`

### 4.2 Best checkpoint 수치

| 지표 | exp10 best | E5 ViT | Δ |
|---|---:|---:|---:|
| val_acc | **0.8608** | 0.8458 | **+1.50%p** |
| Macro F1 | 0.8601 | 0.8470 | +1.31%p |
| NLL | **0.4432** | 0.6977 | −36.5% (낮을수록 우수) |

### 4.3 SWA checkpoint 수치

| 지표 | exp10 SWA | exp10 best | Δ |
|---|---:|---:|---:|
| val_acc | **0.8617** | 0.8608 | +0.09%p |
| Macro F1 | 0.8609 | 0.8601 | +0.08%p |
| NLL | **0.4196** | 0.4432 | −5.3% |

SWA 가 best 보다 일관되게 우위. 특히 NLL 개선이 두드러져 **calibration 이 좋다** 는 SWA 이론과 일치.

### 4.4 앙상블 재탐색 (8 시나리오 exhaustive)

모두 `ensemble_search.py` (DE optimization, maxiter 60~100, bbox crop val) 로 재탐색. predict.py 호환 `weight_opt_raw` 기준.

| 시나리오 | 구성 | val_acc | Δ | Macro F1 | NLL |
|---|---|---:|---:|---:|---:|
| **기존 (제출된 v1.0.0)** | E2+E4+E5+E9 | 0.8692 | — | 0.8697 | 0.4593 |
| A | E2+E4+**swa**+E9 | 0.8625 | −0.67 | 0.8624 | 0.4384 |
| B | E2+E4+**best**+E9 | 0.8575 | −1.17 | 0.8570 | 0.4451 |
| **C (5m)** | E2+E4+E5+**swa**+E9 | **0.8750** | **+0.58** | 0.8751 | 0.4210 |
| **D (4m) ⭐** | E2+E5+**swa**+E9 | **0.8750** | **+0.58** | 0.8751 | 0.4206 |
| F (3m) | E5+**swa**+E9 | 0.8708 | +0.16 | — | — |
| G (2 ViT10) | E2+E5+**best+swa**+E9 | 0.8750 | +0.58 | — | 0.4223 |
| H (6m, all) | E2+E4+E5+**best+swa**+E9 | 0.8750 | +0.58 | — | 0.4214 |

**D 를 메인 제출로 선정** (`models/ensemble_final_4m.json`). 이유:
- 가장 간결 (4 모델) + 최고 점수 달성
- E4 weight 0.034 로 실질 무쓸모 → 제거해도 동일 성능
- exp10 best 추가 (G, H) 도 정확도 동일 → SWA 만 선택

**핵심 관측**:
1. **E5 와 exp10_swa 는 상호보완** — 한쪽만 있으면 0.86 대 (A, B). 둘 다 있어야 0.8750 (C, D, G, H).
2. **exp10 best vs SWA** — SWA 가 앙상블에서도 우위. calibration 좋음 (NLL 낮음) → weighted voting 에 유리.
3. **DE weights (D 시나리오)** — E2: 0.098 / E5: 0.276 / exp10_swa: 0.355 / E9: 0.270. exp10 이 최대 기여.

---

## 5. 판정

### 5.1 단독 모델 비교

exp10_vit_fullcombo (best) = **0.8608 > E5 (0.8458)**. **확정 개선** (+1.50%p).  
exp10 SWA = **0.8617**. Full combo 효과 검증.

### 5.2 앙상블 ceiling 돌파

신 앙상블 D = **0.8750 > 0.8692**. **기존 앙상블 돌파** (+0.58%p).  
통계적 해석: val 1200 에서 7장 추가 정답. p-hat 차이 기준 noise 범위 이내지만, NLL 이 0.459 → 0.421 로 뚜렷 개선 되어 signal 로 판단.

### 5.3 SigLIP fullcombo 추가 학습 결정

ViT 결과 positive → **SigLIP fullcombo 학습 진행 가치 있음** (3-way loss: KD + soft + hard, 3~4h).  
단, val 천장이 0.8750 에 수렴하는 양상 (6 시나리오 동일) → 추가 member 를 넣어도 유사 plateau 가능. **SigLIP fullcombo 는 다양성 확보보다 E9 의 NLL (0.9226) 개선이 주 목적**.

---

## 6. 파일 산출물

- `scripts/train_vit_fullcombo.py` — 학습 스크립트 (584 줄)
- `results/soft_labels/train_soft.npz` — 3-rater 투표 분포 (5,994 장)
- `results/soft_labels/val_soft.npz` — (1,200 장)
- `models/exp10_vit_fullcombo.pt` — best val_acc checkpoint
- `models/exp10_vit_fullcombo.swa.pt` — SWA averaged checkpoint
- `logs/exp10_vit_fullcombo.csv` — epoch 별 지표
- `logs/exp10_vit_fullcombo.meta.json` — 설정 + best 수치
- `reports/fullcombo_experiment.md` — 본 문서

---

## 7. 부록

### 7.1 Mixup 구현 (timm 안 쓰고 수동)

```python
def mixup_batch(x, soft, hard, alpha, prob, rng):
    if alpha <= 0 or rng.random() > prob:
        return x, soft, hard, 1.0
    lam = float(rng.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    soft_mix = lam * soft + (1.0 - lam) * soft[idx]
    hard_mix = lam * hard + (1.0 - lam) * hard[idx]
    return x_mix, soft_mix, hard_mix, lam
```

### 7.2 감정 매핑 (한국어 faceExp → 4-class)

- **anger** ← {분노, 짜증, 화남, 격분}
- **happy** ← {기쁨, 행복, 즐거움, 만족}
- **panic** ← {놀람, 공포, 불안, 당황, 두려움}
- **sadness** ← {슬픔, 우울, 상처, 허탈}
- 매핑 외 값 (중립 등) → 해당 라벨러 1표 skip
- 3인 모두 skip → GT one-hot fallback (34 건)

---

# PART 2 — 90% 돌파 반격 작전 (2026-04-23 개시)

## 8. 3조 대비 열세 인정 (KST 2026-04-23)

3조 공개 수치 (노이즈 472장 포함 상태) 가 우리보다 앞섬:

| 팀 | 최고 단독 | 방식 |
|---|---:|---|
| **3조 EfficientNetV2-M 단독** | **88.54%** | 32 epoch, IN21k_ft_IN1k |
| 3조 MaxViT-B | 87.97% | - |
| 3조 ConvNeXt-L | 87.39% | - |
| 우리 exp11 DINOv3-B SWA 단독 | 86.33% | 20 epoch, LVD-1689M |
| 우리 앙상블 I (5m) | **87.58%** | - |

→ **3조 단독 > 우리 앙상블 (−0.96%p)**. 찐 패배 상태.

### 원인 진단 (Agent 딥서치 기반)

| 요인 | 우리 | 3조 | 영향 |
|---|---|---|:-:|
| 백본 체급 | Base 86M only | Large 계열 (196~303M) 포함 | 🔴 원인 70% |
| 해상도 | 224 | 384 (EffV2-M default) | 🔴 +1~2%p |
| 학습 epoch | 20 | 32 | 🟡 +0.3~0.5%p |
| CNN inductive bias | ViT 위주 | EffV2/ConvNeXt 위주 | 🟡 소규모 데이터 7200 에 CNN 유리 |

**우리 trick stack (soft + mixup + SWA) 은 유효 asset**. Large 백본 + 해상도 384 로 올리고 trick 유지하면 **+2~3%p 추가 가능**. 3조 이길 수 있다.

## 9. 반격 실험 로드맵 (D-7 ~ D-1)

### D-7 (2026-04-23, 오늘)
- **exp12**: `tf_efficientnetv2_m.in21k_ft_in1k` @384, 30ep, fullcombo — **3조 모델 재현 + 우리 레시피**. 목표 89~91%. 진행 중 🔄

### D-6 (2026-04-24)
- **exp13**: `convnext_large.fb_in22k_ft_in1k_384` @384, 25ep, fullcombo **+ CutMix + RandAugment**. 목표 88~90%.

### D-5 (2026-04-25)
- **exp14**: `vit_large_patch16_dinov3.lvd1689m` @224 (또는 256), 25ep, fullcombo. 목표 88~89.5%.

### D-4 (2026-04-26)
- **exp15** (단타 킥): `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` @448, 20ep, fullcombo. 목표 89~90%. IN1k top1 90.06% 의 전이.

### D-3 ~ D-2 (2026-04-27~28)
- **앙상블 재탐색**: exp12/13/14/15 + 기존 best (exp11 DINOv3-B, E2 ResNet, E9 SigLIP-KD) → Forward Stepwise + Weighted BMA.
- **TTA**: hflip + 3-scale (224/256/288) + bbox vs MTCNN.
- 목표: **90.0~91.0%**

### D-1 (2026-04-29)
- 제출물 (predict.py smoke + README + experiments.md)
- 최종 commit + 백업

## 10. 실험별 실제 실행 명령 (재현용)

### exp12 — EfficientNetV2-M @384 fullcombo (진행 중)
```bash
python scripts/train_vit_fullcombo.py \
  --name exp12_effnetv2_m_fullcombo \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k \
  --img-size 384 \
  --stage-a-epochs 5 --epochs 30 --batch-size 24 \
  --lr-head 1e-3 --lr-backbone 8e-5 \
  --weight-decay 1e-4 --warmup-epochs 2 \
  --alpha-kl 0.5 --mixup-alpha 0.2 --mixup-prob 0.5 \
  --mixup-start-ep 5 --swa-start-ep 18 \
  --grad-clip 1.0 --min-lr-ratio 0.01 \
  --seed 42 --num-workers 4 --amp bf16 --crop
```
결과: _(학습 완료 후 기재)_

### exp13~15 — 미리 짜놓은 커맨드 (CutMix + RandAugment 적용)
```bash
# exp13 ConvNeXt-Large @384
python scripts/train_vit_fullcombo.py \
  --name exp13_convnext_l_384_fullcombo \
  --backbone convnext_large.fb_in22k_ft_in1k_384 \
  --img-size 384 \
  --stage-a-epochs 3 --epochs 25 --batch-size 16 \
  --lr-head 1e-3 --lr-backbone 5e-5 \
  --weight-decay 5e-2 --warmup-epochs 2 \
  --alpha-kl 0.5 \
  --mixup-alpha 0.2 --mixup-prob 0.5 \
  --cutmix-alpha 1.0 --cutmix-switch-prob 0.5 \
  --rand-augment --ra-num-ops 2 --ra-magnitude 7 \
  --mixup-start-ep 3 --swa-start-ep 15 \
  --grad-clip 1.0 --min-lr-ratio 0.01 \
  --seed 42 --num-workers 4 --amp bf16 --crop

# exp14 DINOv3 ViT-L @256
python scripts/train_vit_fullcombo.py \
  --name exp14_dinov3_l_256_fullcombo \
  --backbone vit_large_patch16_dinov3 \
  --img-size 256 \
  --stage-a-epochs 5 --epochs 25 --batch-size 16 \
  --lr-head 1e-3 --lr-backbone 3e-5 \
  --weight-decay 1e-4 --warmup-epochs 2 \
  --alpha-kl 0.5 \
  --mixup-alpha 0.2 --mixup-prob 0.5 \
  --cutmix-alpha 1.0 --cutmix-switch-prob 0.5 \
  --rand-augment --ra-num-ops 2 --ra-magnitude 7 \
  --mixup-start-ep 5 --swa-start-ep 18 \
  --grad-clip 1.0 --min-lr-ratio 0.01 \
  --seed 42 --num-workers 4 --amp bf16 --crop

# exp15 EVA-02 Large @448
python scripts/train_vit_fullcombo.py \
  --name exp15_eva02_l_448_fullcombo \
  --backbone eva02_large_patch14_448.mim_m38m_ft_in22k_in1k \
  --img-size 448 \
  --stage-a-epochs 3 --epochs 20 --batch-size 8 \
  --lr-head 1e-3 --lr-backbone 3e-5 \
  --weight-decay 5e-2 --warmup-epochs 2 \
  --alpha-kl 0.5 \
  --mixup-alpha 0.2 --mixup-prob 0.5 \
  --cutmix-alpha 1.0 --cutmix-switch-prob 0.5 \
  --rand-augment --ra-num-ops 2 --ra-magnitude 9 \
  --mixup-start-ep 3 --swa-start-ep 12 \
  --grad-clip 1.0 --min-lr-ratio 0.01 \
  --seed 42 --num-workers 4 --amp bf16 --crop
```

## 11. 스크립트 강화 기록 (2026-04-23)

`scripts/train_vit_fullcombo.py` 에 추가된 기능:

- `--backbone` / `--img-size` CLI 인자 (exp11 부터 사용)
- timm 범용 `set_backbone_trainable` — `get_classifier()` 동적 감지, `head`/`classifier`/`fc.` 이름 fallback
- predict.py meta 저장 시 `model_name` / `img_size` 포함
- SWA AveragedModel 저장 시 `module.` prefix strip (exp10_swa 버그 fix)
- **CutMix 내부 구현** (soft label 호환) + `mixup_batch` 통합
- **RandAugment** CLI 옵션 (torchvision) — exp13 부터 적용
- argparse help format bug fix (`%%` escape)

## 12. 버그 박멸 기록 (누적)

| # | 날짜 | 버그 | Fix |
|:-:|---|---|---|
| 1 | 04-18 | rotate_mask_np ori=7 TRANSVERSE 계산 오류 | scripts/ 수정 + Agent QA 검증 |
| 2 | 04-18 | normalize_orientation PNG 저장 이슈 | JPEG 강제 |
| 3 | 04-19 | eda.py 한글 폰트/axis off 13 이슈 | koreanize-matplotlib |
| 4 | 04-22 | SWA AveragedModel 저장 시 `module.` prefix 로 predict.py 가 random init → acc 0.23 | `swa_model.module.state_dict()` 로 inner SD 만 저장 |
| 5 | 04-22 | predict.py 의 `_load_torch_vit` 가 DINOv3 특수 param (`reg_token`, `gamma_1/2`) 인식 못 해 25 unexpected keys → acc 0.25 | `args.backbone` + `img_size` fallback 체인 + try/except TypeError |
| 6 | 04-22 | train_vit_fullcombo meta.json 에 best_val_f1/nll 빠짐 | best_f1/best_nll 트래커 추가 + meta 기록 |
| 7 | 04-23 | argparse help format error (`50%)` 가 format string 해석 시도) | `50%%` escape |

## 13. AIHub self-training 데이터 조사 (04-22)

**1순위 GO**: AIHub "한국인 감정인식을 위한 복합 영상" (dataSetSn=82)
- ~500,000장 (우리 4감정 필터 후 ~280,000)
- 감정 매핑 완벽 (기쁨→happy, 분노→anger, 당황→panic, 슬픔→sadness)
- 승인 1~3일, 내국인 한정

**정원 action 필요** — AIHub 가입 + 신청. 승인 대기 중 exp13~15 병렬. 승인되면 pseudo-label self-training 으로 추가 +1~2%p.

2~5순위 모두 NO-GO (도메인 미스매치 or 데이터 형식 부적합).


---

# PART 3 — 플랜 v2 최종 확정 (3조 겹침 회피, SSOT 검증, 2026-04-23)

## 14. 3조 결과 허점 분석

3조 GPT 분석 원본 vs 우리 재분석:

| 3조 주장 | 우리 반박 |
|---|---|
| EfficientNetV2-M 88.54% = 1위 | Accuracy 만. **Top-2 Accuracy 는 6위 (95.99%), ROC-AUC 는 2위 (0.9734)** |
| 7 모델 체계적 비교 | **학습 epoch 불공정 (32 vs 10~14 혼합)** — ConvNeXt-Base 10ep 로 꼴찌 산출 |
| "88.54% 가 최고" | **앙상블 없음**. 우리 앙상블 87.58% > 3조 2~7위 전부 |
| 노이즈 472장 환경 | Cohen's Kappa 0.8472 (almost perfect) → **val 에는 노이즈 영향 작음** |
| 단독 모델 비교 | soft label / Mixup / SWA / TTA / SSL pretrain / self-training **전부 언급 없음** |

## 15. 플랜 v2 — "3조와 안 겹치는 경기장" 차별화

### 차별화 축 (3조 못 한 것)

| # | 축 | 우리만 가진 것 |
|:-:|---|---|
| 1 | **앙상블** | 4~6 member weighted DE 최적화 |
| 2 | **3-rater Soft Label** | KL(soft) + KL(hard) 혼합 |
| 3 | **Mixup + CutMix** | soft label 호환 수동 구현 |
| 4 | **SWA (Stochastic Weight Averaging)** | ep12~20 평균 ckpt |
| 5 | **RandAugment** | torchvision, exp13 부터 |
| 6 | **SSL pretrain 백본** | EVA-02 / DINOv3 / SigLIP2 (MIM/SSL/contrastive) |
| 7 | **TTA** | hflip/5crop/multi-scale 조합 |
| 8 | **Self-training** | AIHub K-Face 28만 pseudo-label (승인 대기) |
| 9 | **Knowledge Distillation** | 앙상블→student (시간 여유 시) |

### 실험 확정표 (exp13~16)

| Exp | timm id | Pretrain | 파람 | 해상도 | IN1k top1 | 기대 val | 3조 겹침 |
|:-:|---|---|---:|---:|---:|:-:|:-:|
| **exp13** | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | MIM+MAE (m38m) | 304M | 448 | **90.06** | **89~90%** | 🟢 NO |
| **exp14** | `vit_large_patch16_dinov3.lvd1689m` | SSL LVD-1689M | 303M | 512 | ~87~88 | 88~89% | 🟢 NO |
| **exp15** | `vit_large_patch16_siglip_384.v2_webli` | contrastive text-image | ~300M | 384 | — | 88~90% | 🟢 NO |
| **exp16** (optional) | FaRL ViT-B/16 (수동 로드) | 얼굴 전용 contrastive | 86M | 448 | — | 87~88% | 🟢 NO + 도메인 적합 |

## 16. Skip 확정 모델 (근거)

| 모델 | 이유 |
|---|---|
| Sapiens | CC-BY-NC (비상용 라이선스), 1024 해상도 얼굴 crop 오버킬 |
| DINOv2 ViT-L @518 | DINOv3 가 상위호환 (Meta 공식) |
| BEiTv2-L | MIM 계열은 EVA-02 (IN1k 90.06) 가 더 강함 |
| EfficientNetV2-L | 3조 계열 겹침 회피 |
| MaxViT-L / Swin-L / ConvNeXt-L | 3조 계열 겹침 회피 |

## 17. DataLoader 최적화 기록 (2026-04-23)

exp12 첫 번째 run 에서 GPU util 0% 목격 → num_workers 4 가 원인. 수정:

- `num_workers` default 4 → **16** (CPU 96 cores 활용)
- `persistent_workers=True` (epoch 재생성 오버헤드 제거)
- `prefetch_factor=4` (워커당 선로딩)
- `batch_size` 24 → 32 (VRAM 45GB 여유 활용)

**exp12 재시작 결과 (05:28 KST)**: GPU 94% 활용 확인. epoch 당 시간 ~70~90s 예상.
