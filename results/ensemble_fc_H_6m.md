# Ensemble Search Report

- val samples: **1200** (data_rot/img/val)
- seed: 42
- de_maxiter: 100
- apply_ts (final json): False
- skip_stacking: False

## Per-model val metrics (no ensemble)

| # | model | val_acc | val_macro_f1 | val_nll | T* (NLL opt) |
|:-:|---|---:|---:|---:|---:|
| 0 | `exp02_resnet50_ft_crop_aug` | 0.7758 | 0.7747 | 0.6362 | 1.034 |
| 1 | `exp04_effnet_ft_balanced` | 0.7933 | 0.7940 | 0.5666 | 1.063 |
| 2 | `exp05_vit_b16_two_stage` | 0.8383 | 0.8397 | 0.5223 | 1.073 |
| 3 | `exp10_vit_fullcombo` | 0.8317 | 0.8320 | 0.5144 | 1.275 |
| 4 | `exp10_vit_fullcombo.swa` | 0.8442 | 0.8444 | 0.4737 | 1.217 |
| 5 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8600 | 0.8598 | 0.4322 | M=6 균등 평균 |
| ts_uniform_soft_voting | 0.8525 | 0.8524 | 0.4438 | TS per-model → uniform avg |
| weight_opt_raw | 0.8750 | 0.8751 | 0.4214 | DE on raw probs, w=[0.123, 0.004, 0.259, 0.005, 0.358, 0.252] |
| weight_opt_ts | 0.8733 | 0.8736 | 0.4270 | DE on TS probs, w=[0.074, 0.004, 0.245, 0.13, 0.233, 0.315] |
| stacking_lr_oof | 0.8642 | 0.8642 | 0.3935 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp04_effnet_ft_balanced | 0.7933 | 0.7940 | 0.5666 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8383 | 0.8397 | 0.5223 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo | 0.8317 | 0.8320 | 0.5144 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo.swa | 0.8442 | 0.8444 | 0.4737 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_raw**
- val_acc = 0.8750
- val_macro_f1 = 0.8751
- val_nll = 0.4214
- weights = [0.1228, 0.0035, 0.2586, 0.0048, 0.3585, 0.2518]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
