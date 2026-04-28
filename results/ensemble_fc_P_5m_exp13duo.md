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
| 1 | `exp05_vit_b16_two_stage` | 0.8383 | 0.8397 | 0.5223 | 1.073 |
| 2 | `exp13_eva02_l_448_fullcombo` | 0.8483 | 0.8500 | 0.4594 | 1.058 |
| 3 | `exp13_eva02_l_448_fullcombo.swa` | 0.8483 | 0.8483 | 0.4977 | 1.546 |
| 4 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8642 | 0.8643 | 0.4146 | M=5 균등 평균 |
| ts_uniform_soft_voting | 0.8642 | 0.8644 | 0.4264 | TS per-model → uniform avg |
| weight_opt_raw | 0.8750 | 0.8754 | 0.4078 | DE on raw probs, w=[0.115, 0.183, 0.347, 0.132, 0.223] |
| weight_opt_ts | 0.8742 | 0.8747 | 0.4269 | DE on TS probs, w=[0.156, 0.332, 0.225, 0.118, 0.169] |
| stacking_lr_oof | 0.8642 | 0.8644 | 0.3751 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8383 | 0.8397 | 0.5223 | 단일 모델 기준점 |
| single:exp13_eva02_l_448_fullcombo | 0.8483 | 0.8500 | 0.4594 | 단일 모델 기준점 |
| single:exp13_eva02_l_448_fullcombo.swa | 0.8483 | 0.8483 | 0.4977 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_raw**
- val_acc = 0.8750
- val_macro_f1 = 0.8754
- val_nll = 0.4078
- weights = [0.1149, 0.1831, 0.3468, 0.132, 0.2232]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
