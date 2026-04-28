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
| 2 | `exp11_dinov3_fullcombo.swa` | 0.8517 | 0.8528 | 0.4482 | 1.221 |
| 3 | `exp13_eva02_l_448_fullcombo.swa` | 0.8483 | 0.8483 | 0.4977 | 1.546 |
| 4 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8692 | 0.8692 | 0.4136 | M=5 균등 평균 |
| ts_uniform_soft_voting | 0.8667 | 0.8668 | 0.4249 | TS per-model → uniform avg |
| weight_opt_raw | 0.8767 | 0.8770 | 0.4045 | DE on raw probs, w=[0.116, 0.199, 0.405, 0.18, 0.101] |
| weight_opt_ts | 0.8767 | 0.8770 | 0.4124 | DE on TS probs, w=[0.098, 0.178, 0.393, 0.163, 0.168] |
| stacking_lr_oof | 0.8725 | 0.8727 | 0.3710 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8383 | 0.8397 | 0.5223 | 단일 모델 기준점 |
| single:exp11_dinov3_fullcombo.swa | 0.8517 | 0.8528 | 0.4482 | 단일 모델 기준점 |
| single:exp13_eva02_l_448_fullcombo.swa | 0.8483 | 0.8483 | 0.4977 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_raw**
- val_acc = 0.8767
- val_macro_f1 = 0.8770
- val_nll = 0.4045
- weights = [0.1157, 0.1991, 0.4048, 0.1797, 0.1007]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
