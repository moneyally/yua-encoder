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
| 2 | `exp05_vit_b16_two_stage` | 0.8383 | 0.8397 | 0.5223 | 1.073 |
| 3 | `exp06_siglip_linear_probe` | 0.8242 | 0.8231 | 0.5069 | 0.907 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8433 | 0.8435 | 0.4726 | M=4 균등 평균 |
| ts_uniform_soft_voting | 0.8433 | 0.8434 | 0.4759 | TS per-model → uniform avg |
| weight_opt_raw | 0.8600 | 0.8603 | 0.4555 | DE on raw probs, w=[0.042, 0.107, 0.382, 0.469] |
| weight_opt_ts | 0.8600 | 0.8603 | 0.4557 | DE on TS probs, w=[0.003, 0.108, 0.435, 0.454] |
| stacking_lr_oof | 0.8567 | 0.8569 | 0.4242 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp04_effnet_ft_balanced | 0.7933 | 0.7940 | 0.5666 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8383 | 0.8397 | 0.5223 | 단일 모델 기준점 |
| single:exp06_siglip_linear_probe | 0.8242 | 0.8231 | 0.5069 | 단일 모델 기준점 |

## Best method

- **weight_opt_raw**
- val_acc = 0.8600
- val_macro_f1 = 0.8603
- val_nll = 0.4555
- weights = [0.0423, 0.1069, 0.3815, 0.4693]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
