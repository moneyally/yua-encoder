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
| 1 | `exp04_effnet_ft_balanced` | 0.7933 | 0.7940 | 0.5666 | 1.063 |
| 2 | `exp05_vit_b16_two_stage` | 0.8383 | 0.8397 | 0.5223 | 1.073 |
| 3 | `exp10_vit_fullcombo.swa` | 0.8442 | 0.8444 | 0.4737 | 1.217 |
| 4 | `exp09_siglip_kd_tsoff_T4_a07_uf4` | 0.8333 | 0.8331 | 0.9226 | 3.323 |

## Ensemble methods

| method | val_acc | val_macro_f1 | val_nll | notes |
|---|---:|---:|---:|---|
| uniform_soft_voting | 0.8558 | 0.8558 | 0.4381 | M=5 균등 평균 |
| ts_uniform_soft_voting | 0.8542 | 0.8542 | 0.4488 | TS per-model → uniform avg |
| weight_opt_raw | 0.8750 | 0.8751 | 0.4210 | DE on raw probs, w=[0.122, 0.034, 0.221, 0.361, 0.262] |
| weight_opt_ts | 0.8742 | 0.8744 | 0.4237 | DE on TS probs, w=[0.058, 0.021, 0.241, 0.356, 0.325] |
| stacking_lr_oof | 0.8642 | 0.8644 | 0.3935 | 참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외 |
| single:exp02_resnet50_ft_crop_aug | 0.7758 | 0.7747 | 0.6362 | 단일 모델 기준점 |
| single:exp04_effnet_ft_balanced | 0.7933 | 0.7940 | 0.5666 | 단일 모델 기준점 |
| single:exp05_vit_b16_two_stage | 0.8383 | 0.8397 | 0.5223 | 단일 모델 기준점 |
| single:exp10_vit_fullcombo.swa | 0.8442 | 0.8444 | 0.4737 | 단일 모델 기준점 |
| single:exp09_siglip_kd_tsoff_T4_a07_uf4 | 0.8333 | 0.8331 | 0.9226 | 단일 모델 기준점 |

## Best method

- **weight_opt_raw**
- val_acc = 0.8750
- val_macro_f1 = 0.8751
- val_nll = 0.4210
- weights = [0.1219, 0.0345, 0.2214, 0.3607, 0.2616]

## Notes

- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 (증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).
- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).
- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.
- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.
