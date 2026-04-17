# Ensemble Search Report

- val samples: **1200** (data_rot/img/val)
- seed: 42
- de_maxiter: 60
- apply_ts (final json): False
- skip_stacking: False

## Per-model val metrics (no ensemble)

| # | model | val_acc | val_macro_f1 | val_nll | T* (NLL opt) |
|:-:|---|---:|---:|---:|---:|
| 0 | `exp02_resnet50_ft_crop_aug` | 0.7450 | 0.7449 | 0.6948 | 0.871 |
| 1 | `exp04_effnet_ft_balanced` | 0.7792 | 0.7823 | 0.6026 | 0.992 |
| 2 | `exp05_vit_b16_two_stage` | 0.8258 | 0.8285 | 0.5458 | 1.089 |
| 3 | `exp06_siglip_linear_probe` | 0.7942 | 0.7943 | 0.5557 | 0.715 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8325 | 0.8339 | 0.5135 | M=4 균등 평균 |
| ts_uniform_soft_voting | 0.8292 | 0.8305 | 0.4992 | TS per-model → uniform avg |
| weight_opt_raw | 0.8442 | 0.8458 | 0.5023 | DE on raw probs, w=[0.134, 0.115, 0.287, 0.463] |
| weight_opt_ts | 0.8450 | 0.8467 | 0.4818 | DE on TS probs, w=[0.128, 0.097, 0.454, 0.321] |
| stacking_lr_oof | 0.8517 | 0.8527 | 0.4235 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7450 | 0.7449 | 0.6948 | 단일 모델 기준점 |
| single:exp04_effnet_ft_balanced | 0.7792 | 0.7823 | 0.6026 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8258 | 0.8285 | 0.5458 | 단일 모델 기준점 |
| single:exp06_siglip_linear_probe | 0.7942 | 0.7943 | 0.5557 | 단일 모델 기준점 |

## Best method

- **weight_opt_ts**
- val_acc = 0.8450
- val_macro_f1 = 0.8467
- val_nll = 0.4818
- weights = [0.1344, 0.1151, 0.287, 0.4635]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
