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
| 2 | `exp10_vit_fullcombo` | 0.8317 | 0.8320 | 0.5144 | 1.275 |
| 3 | `exp10_vit_fullcombo.swa` | 0.8442 | 0.8444 | 0.4737 | 1.217 |
| 4 | `exp11_dinov3_fullcombo.swa` | 0.8517 | 0.8528 | 0.4482 | 1.221 |
| 5 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8650 | 0.8652 | 0.4186 | M=6 균등 평균 |
| ts_uniform_soft_voting | 0.8642 | 0.8642 | 0.4295 | TS per-model → uniform avg |
| weight_opt_raw | 0.8758 | 0.8760 | 0.4226 | DE on raw probs, w=[0.211, 0.188, 0.001, 0.322, 0.08, 0.198] |
| weight_opt_ts | 0.8758 | 0.8759 | 0.4205 | DE on TS probs, w=[0.069, 0.22, 0.003, 0.283, 0.084, 0.342] |
| stacking_lr_oof | 0.8733 | 0.8735 | 0.3704 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8383 | 0.8397 | 0.5223 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo | 0.8317 | 0.8320 | 0.5144 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo.swa | 0.8442 | 0.8444 | 0.4737 | 단일 모델 기준점 |
| single:exp11_dinov3_fullcombo.swa | 0.8517 | 0.8528 | 0.4482 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_ts**
- val_acc = 0.8758
- val_macro_f1 = 0.8759
- val_nll = 0.4205
- weights = [0.2108, 0.1875, 0.0013, 0.3218, 0.0804, 0.1983]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
