# Ablation Results — `exp05_vit_b16_two_stage.pt`

val set: 1200 images × 12 configs

## Accuracy / F1 / NLL / Latency 요약

| # | crop | TTA | accuracy | macro F1 | NLL | ms/img (p95) |
|---|---|---|---:|---:|---:|---:|
| 1 | none | none | 0.6025 | 0.6023 | 0.9918 | 33.0 (49.8) |
| 2 | none | hflip | 0.6067 | 0.6060 | 0.9786 | 67.7 (108.6) |
| 3 | none | 5crop | 0.2717 | 0.2493 | 1.4173 | 59.1 (63.5) |
| 4 | none | 5crop_mscale | 0.5567 | 0.5621 | 1.0310 | 157.0 (188.3) |
| 5 | bbox | none | 0.8408 | 0.8420 | 0.5235 | 10.3 (13.8) |
| 6 | bbox | hflip | 0.8433 | 0.8448 | 0.5095 | 20.3 (28.3) |
| 7 | bbox | 5crop | 0.2717 | 0.2269 | 1.4288 | 55.9 (56.8) |
| 8 | bbox | 5crop_mscale | 0.7800 | 0.7761 | 0.6622 | 118.9 (131.9) |
| 9 | mtcnn | none | 0.8258 | 0.8285 | 0.5458 | 261.0 (512.7) |
| 10 | mtcnn | hflip | 0.8342 | 0.8369 | 0.5244 | 522.2 (1053.5) |
| 11 | mtcnn | 5crop | 0.2708 | 0.2476 | 1.4173 | 137.4 (160.1) |
| 12 | mtcnn | 5crop_mscale | 0.7633 | 0.7662 | 0.6803 | 481.5 (550.3) |

**Best**: `bbox` + `hflip` → accuracy **0.8433**, F1 **0.8448**

## Per-class F1 — best config

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| anger | 0.7343 | 0.8200 | 0.7748 | 300 |
| happy | 0.9791 | 0.9367 | 0.9574 | 300 |
| panic | 0.8672 | 0.7833 | 0.8231 | 300 |
| sadness | 0.8143 | 0.8333 | 0.8237 | 300 |

## Confusion matrix — best config (rows=gt, cols=pred)

```
          anger     happy     panic   sadness
 anger      246         1        19        34
 happy        7       281         8         4
 panic       43         3       235        19
sadness       39         2         9       250
```

