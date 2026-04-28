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
| 1 | `exp05_vit_b16_two_stage` | 0.8383 | 0.8397 | 0.5223 | 1.073 |
| 2 | `exp10_vit_fullcombo.swa` | 0.8442 | 0.8444 | 0.4737 | 1.217 |
| 3 | `exp11_dinov3_fullcombo.swa` | 0.8517 | 0.8528 | 0.4482 | 1.221 |
| 4 | `exp13_eva02_l_448_fullcombo` | 0.8483 | 0.8500 | 0.4594 | 1.058 |
| 5 | `exp13_eva02_l_448_fullcombo.swa` | 0.8483 | 0.8483 | 0.4977 | 1.546 |
| 6 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8683 | 0.8685 | 0.4040 | M=7 균등 평균 |
| ts_uniform_soft_voting | 0.8683 | 0.8686 | 0.4161 | TS per-model → uniform avg |
| weight_opt_raw | 0.8767 | 0.8769 | 0.4190 | DE on raw probs, w=[0.227, 0.192, 0.264, 0.062, 0.068, 0.051, 0.136] |
| weight_opt_ts | 0.8750 | 0.8753 | 0.4094 | DE on TS probs, w=[0.074, 0.194, 0.023, 0.166, 0.248, 0.153, 0.141] |
| stacking_lr_oof | 0.8725 | 0.8727 | 0.3611 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8383 | 0.8397 | 0.5223 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo.swa | 0.8442 | 0.8444 | 0.4737 | 단일 모델 기준점 |
| single:exp11_dinov3_fullcombo.swa | 0.8517 | 0.8528 | 0.4482 | 단일 모델 기준점 |
| single:exp13_eva02_l_448_fullcombo | 0.8483 | 0.8500 | 0.4594 | 단일 모델 기준점 |
| single:exp13_eva02_l_448_fullcombo.swa | 0.8483 | 0.8483 | 0.4977 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_raw**
- val_acc = 0.8767
- val_macro_f1 = 0.8769
- val_nll = 0.4190
- weights = [0.2273, 0.192, 0.2639, 0.0619, 0.0682, 0.051, 0.1356]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
