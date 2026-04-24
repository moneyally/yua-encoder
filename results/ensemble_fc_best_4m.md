# Ensemble Search Report

- val samples: **1200** (data_rot/img/val)
- seed: 42
- de_maxiter: 60
- apply_ts (final json): False
- skip_stacking: False

## Per-model val metrics (no ensemble)

| # | model | val_acc | val_macro_f1 | val_nll | T* (NLL opt) |
|:-:|---|---:|---:|---:|---:|
| 0 | `exp02_resnet50_ft_crop_aug` | 0.7758 | 0.7747 | 0.6362 | 1.034 |
| 1 | `exp04_effnet_ft_balanced` | 0.7933 | 0.7940 | 0.5666 | 1.063 |
| 2 | `exp10_vit_fullcombo` | 0.8317 | 0.8320 | 0.5144 | 1.275 |
| 3 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8417 | 0.8412 | 0.4571 | M=4 균등 평균 |
| ts_uniform_soft_voting | 0.8350 | 0.8344 | 0.4668 | TS per-model → uniform avg |
| weight_opt_raw | 0.8567 | 0.8563 | 0.4463 | DE on raw probs, w=[0.101, 0.196, 0.542, 0.161] |
| weight_opt_ts | 0.8575 | 0.8570 | 0.4451 | DE on TS probs, w=[0.094, 0.105, 0.469, 0.332] |
| stacking_lr_oof | 0.8575 | 0.8573 | 0.4132 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp04_effnet_ft_balanced | 0.7933 | 0.7940 | 0.5666 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo | 0.8317 | 0.8320 | 0.5144 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_ts**
- val_acc = 0.8575
- val_macro_f1 = 0.8570
- val_nll = 0.4451
- weights = [0.1012, 0.1962, 0.5419, 0.1607]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
