# Ablation Results — `ensemble_final_4m.json`

val set: 1200 images × 4 configs

## Accuracy / F1 / NLL / Latency 요약

| # | crop | TTA | accuracy | macro F1 | NLL | ms/img (p95) |
|---|---|---|---:|---:|---:|---:|
| 1 | bbox | none | 0.8733 | 0.8733 | 0.4211 | 239.8 (261.5) |
| 2 | bbox | hflip | 0.8708 | 0.8708 | 0.4164 | 446.7 (550.4) |
| 3 | mtcnn | none | 0.8583 | 0.8589 | 0.4303 | 493.7 (850.7) |
| 4 | mtcnn | hflip | 0.8583 | 0.8588 | 0.4246 | 1025.1 (1693.4) |

**Best**: `bbox` + `none` → accuracy **0.8733**, F1 **0.8733**

## Per-class F1 — best config

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| anger | 0.8249 | 0.8167 | 0.8208 | 300 |
| happy | 0.9635 | 0.9667 | 0.9651 | 300 |
| panic | 0.8254 | 0.8667 | 0.8455 | 300 |
| sadness | 0.8815 | 0.8433 | 0.8620 | 300 |

## Confusion matrix — best config (rows=gt, cols=pred)

```
          anger     happy     panic   sadness
 anger      245         2        31        22
 happy        2       290         7         1
 panic       24         5       260        11
sadness       26         4        17       253
```

