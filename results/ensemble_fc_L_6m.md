# Ensemble Search Report

- val samples: **1200** (data_rot/img/val)
- seed: 42
- de_maxiter: 150
- apply_ts (final json): False
- skip_stacking: False

## Per-model val metrics (no ensemble)

| # | model | val_acc | val_macro_f1 | val_nll | T* (NLL opt) |
|:-:|---|---:|---:|---:|---:|
| 0 | `exp02_resnet50_ft_crop_aug` | 0.7758 | 0.7747 | 0.6362 | 1.034 |
| 1 | `exp04_effnet_ft_balanced` | 0.7933 | 0.7940 | 0.5666 | 1.063 |
| 2 | `exp05_vit_b16_two_stage` | 0.8383 | 0.8397 | 0.5223 | 1.073 |
| 3 | `exp10_vit_fullcombo.swa` | 0.8442 | 0.8444 | 0.4737 | 1.217 |
| 4 | `exp11_dinov3_fullcombo.swa` | 0.8517 | 0.8528 | 0.4482 | 1.221 |
| 5 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8650 | 0.8651 | 0.4217 | M=6 균등 평균 |
| ts_uniform_soft_voting | 0.8583 | 0.8584 | 0.4343 | TS per-model → uniform avg |
| weight_opt_raw | 0.8758 | 0.8760 | 0.4201 | DE on raw probs, w=[0.212, 0.002, 0.21, 0.253, 0.151, 0.172] |
| weight_opt_ts | 0.8758 | 0.8759 | 0.4208 | DE on TS probs, w=[0.065, 0.008, 0.218, 0.3, 0.069, 0.339] |
| stacking_lr_oof | 0.8733 | 0.8734 | 0.3713 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp04_effnet_ft_balanced | 0.7933 | 0.7940 | 0.5666 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8383 | 0.8397 | 0.5223 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo.swa | 0.8442 | 0.8444 | 0.4737 | 단일 모델 기준점 |
| single:exp11_dinov3_fullcombo.swa | 0.8517 | 0.8528 | 0.4482 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_raw**
- val_acc = 0.8758
- val_macro_f1 = 0.8760
- val_nll = 0.4201
- weights = [0.2122, 0.0017, 0.2097, 0.2526, 0.1515, 0.1723]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
