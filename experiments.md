| date | name | model | config | epochs | val_acc | val_loss | train_acc | note |
|---|---|---|---|---|---|---|---|---|
| 2026-04-17 05:55 | smoke_resnet50_frozen | resnet50_frozen | bs=16 img=224 lr=0.001 crop=0 aug=0 | epochs=1/1 | val_acc=0.2406 | val_loss=1.5370 | final_train_acc=0.2297 | note=smoke |
| 2026-04-17 05:57 | smoke_crop_check | custom_cnn | bs=8 img=96 lr=0.001 crop=1 aug=0 | epochs=1/1 | val_acc=0.0000 | val_loss=1.4758 | final_train_acc=0.0625 | note=crop_smoke |
| 2026-04-17 06:07 | smoke_gpu | resnet50_frozen | bs=32 img=224 lr=0.001 crop=0 aug=0 | epochs=1/1 | val_acc=0.1750 | val_loss=1.8501 | final_train_acc=0.2688 | note=gpu-smoke |
| 2026-04-17 06:16 | phase0_speed | resnet50_frozen | bs=128 img=224 lr=0.001 crop=0 aug=0 | epochs=1/1 | val_acc=0.3075 | val_loss=1.4751 | final_train_acc=0.2805 | note=phase0-speed-profile |
| 2026-04-17 07:07 | exp01_resnet50_baseline | resnet50_frozen | bs=128 img=224 lr=0.001 crop=0 aug=0 | epochs=10/10 | val_acc=0.4042 | val_loss=1.3132 | final_train_acc=0.4868 | note=E1 baseline: ResNet50 frozen, full 5996/1200 |
| 2026-04-17 07:50 | exp02_resnet50_ft_crop_aug | resnet50_ft | bs=32 img=224 lr=1e-04 crop=1 aug=1 | epochs=10/30 | val_acc=0.7733 | val_loss=0.8422 | final_train_acc=0.9878 | note=E2 retry: conv5 unfreeze + crop + aug + bs32 + ls0.1 + cache-order-fix; killed at ep10, best at ep7 |
| 2026-04-17 08:02 | exp03_adv_adamw_cosine | resnet50_ft | bs=32 img=224 lr=5e-05 crop=1 aug=1 | epochs=14/25 | val_acc=0.7617 | val_loss=0.9941 | final_train_acc=0.9624 | note=E3 adv: AdamW wd1e-4 + cosine+warmup3 + strong aug + dropout0.5 + ls0.2 + monitor val_loss |
| 2026-04-17 08:45 | exp04_effnet_ft_balanced | efficientnet_ft | bs=32 img=224 lr=1e-04 crop=1 aug=1 | epochs=2/20 | val_acc=0.7367 | val_loss=0.8161 | final_train_acc=0.7512 | note=E4 partial: EfficientNet-B0 block6 + AdamW + cosine_warmup2 + aug중간 + ls0.1 + monitor val_acc. Epoch 1이 20분 XLA compile, 잘못 kill. 2 epoch 로 중단 |

---

## 📊 실험 비교 요약 (E1~E8)

### 완료 실험

| # | 이름 | 모델 | 주요 기법 | best val_acc | best val_loss | train_acc | gap | 시간 | 평가 |
|---|---|---|---|---:|---:|---:|---:|---|---|
| E1 | exp01_resnet50_baseline | ResNet50 frozen | FC만 학습, bs128, 10ep | 0.4042 | 1.3132 | 0.4868 | 0.08 | 20분 | baseline (underfit) |
| E2 | exp02_resnet50_ft_crop_aug | ResNet50 ft (conv5) | crop+aug+bs32+lr1e-4+ls0.1 | 0.7733 | 0.8422 | 0.988 | 0.21 | 22분 | 이전 best |
| E3 | exp03_adv_adamw_cosine | ResNet50 ft (conv5) | E2+AdamW wd1e-4+cosine_warmup3+strong aug+dropout0.5+ls0.2+monitor=val_loss | 0.7617 | 0.9941 | 0.9624 | 0.15 | 24분 | overfit 억제 성공, val_acc 미달 |
| **E4** | exp04_effnet_ft_balanced | **EfficientNet-B0 ft (block6)** | 중간 aug(rot12/zoom0.05) + AdamW wd1e-4 + cosine_warmup2 + dropout0.4 + ls0.1 | **0.7958** | **0.7526** | 0.9423 | 0.15 | 29분 | **🏆 새 best (18/20ep EarlyStop)** |

### 예정 실험

| # | 이름 | 모델 | 주요 기법 | 상태 |
|---|---|---|---|---|
| E5 | exp05_vit_b16 | **ViT-B/16** (timm, ImageNet21k) | 2-stage: freeze head 5ep → unfreeze all 15ep | completed val_acc=0.8458 |
| E6 | exp06_siglip | **SigLIP ViT-Base** (yua-encoder wrapper) | freeze + PixelShuffle + projection + linear probe | completed val_acc=0.8192 |
| E7 | exp07_softlabel | ResNet50 ft or ViT | annot_A/B/C 3rater vote → soft label (창의성) | 예정 |
| E8 | exp08_ensemble | **E2 + E4 + E5** soft voting | 최종 제출 | 예정 |

### 핵심 발견 (4/17 기준)

- **E1→E2 +0.37 점프**: face crop + conv5 unfreeze + aug 조합이 결정적.
- **E3 val_acc E2 미달**: monitor=val_loss 로 바꾸니 accuracy 기준 best 못 잡음 (label_smoothing 0.2로 loss calibration 바뀐 것도 영향).
- **E4 새 best**: EfficientNet-B0 의 파라미터 효율 + 중간 aug 조합이 ResNet50 보다 **2.25%p 앞섬**. val_loss 는 0.7526 으로 E2 대비 -0.09.
- **모델별 train-val gap**: E2 0.21 > E3 0.15 ≈ E4 0.15. 즉 E3 의 strong reg 와 E4 의 EfficientNet 은 overfit 억제 효과 동등.
- **val_acc 상한 추정**: 3rater 다수결 일치율 0.9045, 전원일치 0.6108 → 실질 ceiling 0.85~0.90.
- **현재 best 0.7958 은 baseline 기준 +39.2%p**. 목표 라인 0.75 이미 돌파.

### 앙상블 후보 분석

- **E2 (ResNet50 ft)** val 0.7733: CNN 강한 inductive bias, 국소 feature 잘 잡음
- **E4 (EfficientNet-B0 ft)** val 0.7958: 파라미터 효율 + 중간 aug 로 overfit 저항
- **E5 (ViT-B/16)** 예정: 전역 attention, 다른 architecture family
- 3개 모두 독립 트랙 → 앙상블 시 **+2~4%p 기대 (0.82~0.84 목표)**

### 다음 레버 Top 3

1. **ViT-B/16 (E5)** — Transformer 다양성 확보.
2. **앙상블 (E8)** — E2 + E4 + E5 soft voting. 0.82~0.84 목표.
3. **soft label (E7)** — 3rater 불일치를 라벨로 녹여 val_acc 상한 끌어올림. 창의성 점수 직결.
| 2026-04-17 08:56 | exp04_effnet_ft_balanced | efficientnet_ft | bs=32 img=224 lr=0.0001 crop=1 aug=1 | epochs=18/20 | val_acc=0.7958 | val_loss=0.7526 | final_train_acc=0.9423 | note=E4 retry: EfficientNet-B0 ft block6 + AdamW + cosine_warmup2 + aug중간 + ls0.1 (1st epoch 20min XLA compile 예상, kill 금지) |
| 2026-04-17 09:49 | exp05_vit_b16_two_stage | vit_base_patch16_224 | bs=32 img=224 lr=0.001 lr_bb=5e-05 two_stage=1 crop=1 aug=1 | epochs=19/20 | val_acc=0.8458 | val_loss=0.6977 | final_train_acc=0.9826 | note=E5 ViT-B/16 + 2-stage(5A+15B) + crop+aug + bf16 | val_f1=0.8470 |
| 2026-04-17 11:15 | exp06_siglip_linear_probe | siglip_base_p16_384 | bs=32 img=384 lr=0.001 lr_bb=5e-05 two_stage=0 unfreeze=0 crop=1 aug=1 | epochs=10/15 | val_acc=0.8192 | val_loss=0.7338 | final_train_acc=0.9528 | note=E6 SigLIP linear probe (yua-encoder wrapper) + fill=128 + class_weight=auto + SSOT + worker_init | val_f1=0.8183 |
| 2026-04-17 16:23 | exp09_siglip_kd_tsoff_T4_a07_uf0 | siglip_kd_unfreeze0 | bs=48 img=384 T=4.0 α=0.7 unfreeze=0 crop=1 aug=1 | epochs=5/5 | val_acc=0.8308 | val_loss=0.9446 | final_train_acc=0.9197 | note=KD teacher=train_teacher_tsoff.npz save_gate=pass |
| 2026-04-17 16:37 | exp09_siglip_kd_tson_T4_a07_uf0 | siglip_kd_unfreeze0 | bs=48 img=384 T=4.0 α=0.7 unfreeze=0 crop=1 aug=1 | epochs=5/5 | val_acc=0.8292 | val_loss=1.1074 | final_train_acc=0.9181 | note=KD teacher=train_teacher_tson.npz save_gate=pass |
| 2026-04-17 16:50 | exp09_siglip_kd_tsoff_T2_a07_uf0 | siglip_kd_unfreeze0 | bs=48 img=384 T=2.0 α=0.7 unfreeze=0 crop=1 aug=1 | epochs=5/5 | val_acc=0.8267 | val_loss=0.5759 | final_train_acc=0.9277 | note=KD teacher=train_teacher_tsoff.npz save_gate=pass |
| 2026-04-17 17:04 | exp09_siglip_kd_tsoff_T4_a07_uf2 | siglip_kd_unfreeze2 | bs=48 img=384 T=4.0 α=0.7 unfreeze=2 crop=1 aug=1 | epochs=5/5 | val_acc=0.8317 | val_loss=0.9421 | final_train_acc=0.9239 | note=KD teacher=train_teacher_tsoff.npz save_gate=pass |
| 2026-04-17 17:18 | exp09_ce_only_control | siglip_kd_unfreeze0 | bs=48 img=384 T=4.0 α=0.0 unfreeze=0 crop=1 aug=1 | epochs=5/5 | val_acc=0.8267 | val_loss=0.5509 | final_train_acc=0.9336 | note=KD teacher=train_teacher_tsoff.npz save_gate=pass |
| 2026-04-17 17:31 | exp09_siglip_kd_tsoff_T4_a07_uf4 | siglip_kd_unfreeze4 | bs=48 img=384 T=4.0 α=0.7 unfreeze=4 crop=1 aug=1 | epochs=5/5 | val_acc=0.8383 | val_loss=0.9275 | final_train_acc=0.9336 | note=KD teacher=train_teacher_tsoff.npz save_gate=pass |
| 2026-04-22 04:07 | exp10_vit_fullcombo (best) | vit_base_patch16_224 | bs=32 img=224 lr_head=1e-3 lr_bb=5e-5 two_stage=1 α_kl=0.5 mixup=0.2@0.5 swa_start=12 crop=1 aug=1 | epochs=20/20 | val_acc=0.8608 | val_loss=0.4432 | val_f1=0.8601 | note=E10 Full Combo: Soft(3rater)+Mixup+SWA — ViT 단독 +1.50%p vs E5 |
| 2026-04-22 04:08 | exp10_vit_fullcombo (swa) | vit_base_patch16_224 | same as best + SWA averaged | epochs=20/20 | val_acc=0.8617 | val_loss=0.4196 | val_f1=0.8609 | note=E10 SWA averaged ckpt — calibration 우위 (NLL 0.4196 vs best 0.4432) |
| 2026-04-22 04:13 | ensemble_final_4m (D) | E2+E5+exp10_swa+E9 | weights=[0.098,0.276,0.355,0.270] weight_opt_raw DE | — | val_acc=0.8750 | val_loss=0.4206 | val_f1=0.8751 | note=**신 메인 제출**. 기존 0.8692 → 0.8750 (+0.58%p) 돌파. 8 시나리오 exhaustive 탐색 결과 plateau |
