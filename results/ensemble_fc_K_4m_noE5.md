# Ensemble Search Report

- val samples: **1200** (data_rot/img/val)
- seed: 42
- de_maxiter: 80
- apply_ts (final json): False
- skip_stacking: False

## Per-model val metrics (no ensemble)

| # | model | val_acc | val_macro_f1 | val_nll | T* (NLL opt) |
|:-:|---|---:|---:|---:|---:|
| 0 | `exp02_resnet50_ft_crop_aug` | 0.7758 | 0.7747 | 0.6362 | 1.034 |
| 1 | `exp10_vit_fullcombo.swa` | 0.8442 | 0.8444 | 0.4737 | 1.217 |
| 2 | `exp11_dinov3_fullcombo.swa` | 0.8517 | 0.8528 | 0.4482 | 1.221 |
| 3 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8608 | 0.8607 | 0.4273 | M=4 균등 평균 |
| ts_uniform_soft_voting | 0.8625 | 0.8623 | 0.4355 | TS per-model → uniform avg |
| weight_opt_raw | 0.8700 | 0.8703 | 0.4201 | DE on raw probs, w=[0.053, 0.088, 0.543, 0.316] |
| weight_opt_ts | 0.8708 | 0.8710 | 0.4180 | DE on TS probs, w=[0.064, 0.029, 0.497, 0.411] |
| stacking_lr_oof | 0.8725 | 0.8725 | 0.3753 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo.swa | 0.8442 | 0.8444 | 0.4737 | 단일 모델 기준점 |
| single:exp11_dinov3_fullcombo.swa | 0.8517 | 0.8528 | 0.4482 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_ts**
- val_acc = 0.8708
- val_macro_f1 = 0.8710
- val_nll = 0.4180
- weights = [0.053, 0.0876, 0.543, 0.3165]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
