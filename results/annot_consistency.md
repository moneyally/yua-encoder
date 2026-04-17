# 3인 라벨러 일치성 EDA (split=train,val)

## 데이터 요약
- 항목 수: **7194**
- 3인 faceExp 유효: **7194** (skip 0)
- 3인 bbox 유효: **7192** (skip 2)

## 1) faceExp 일치율

- 전원 일치 (3/3): **4394/7194 = 61.08%**
- 다수결 일치 (≥2/3): **6507/7194 = 90.45%**
- 3인 완전 불일치: **687/7194 = 9.55%**

### 클래스별

| 클래스 | 유효 n | 전원일치 % | 다수결 % | 완전불일치 % |
|---|---:|---:|---:|---:|
| anger (분노) | 1800 | 50.50 | 87.28 | 12.72 |
| happy (기쁨) | 1794 | 93.76 | 99.55 | 0.45 |
| panic (당황) | 1800 | 50.94 | 87.17 | 12.83 |
| sadness (슬픔) | 1800 | 49.22 | 87.83 | 12.17 |

### 자주 헷갈리는 감정 쌍 (상위 10)

| 쌍 | 횟수 |
|---|---:|
| 슬픔 ↔ 상처 | 729 |
| 불안 ↔ 분노 | 665 |
| 당황 ↔ 불안 | 580 |
| 슬픔 ↔ 불안 | 550 |
| 슬픔 ↔ 분노 | 441 |
| 당황 ↔ 분노 | 415 |
| 중립 ↔ 당황 | 371 |
| 당황 ↔ 상처 | 333 |
| 상처 ↔ 분노 | 313 |
| 불안 ↔ 상처 | 286 |

## 2) 폴더 클래스(GT) vs 라벨러 faceExp 매칭

| 클래스 | 유효 n | 1명↑ 일치 % | 3명 모두 일치 % |
|---|---:|---:|---:|
| anger (분노) | 1800 | 90.39 | 49.00 |
| happy (기쁨) | 1794 | 99.44 | 93.59 |
| panic (당황) | 1800 | 88.33 | 47.22 |
| sadness (슬픔) | 1800 | 89.39 | 46.22 |

## 3) bbox IoU (annot pair별)

| 쌍 | n | 평균 | 중앙 | 최소 | 최대 | std |
|---|---:|---:|---:|---:|---:|---:|
| A vs B | 7192 | 0.920 | 0.930 | 0.000 | 1.000 | 0.064 |
| A vs C | 7192 | 0.936 | 0.948 | 0.000 | 1.000 | 0.066 |
| B vs C | 7192 | 0.925 | 0.937 | 0.000 | 1.000 | 0.063 |

### 클래스별 평균 IoU (3쌍 평균)

| 클래스 | n | 평균 | 중앙 | 최소 |
|---|---:|---:|---:|---:|
| anger | 1800 | 0.926 | 0.933 | 0.299 |
| happy | 1794 | 0.928 | 0.937 | 0.314 |
| panic | 1799 | 0.928 | 0.932 | 0.576 |
| sadness | 1799 | 0.926 | 0.935 | 0.302 |

## 4) 평균 IoU 최저 top-10 (가장 논쟁적 bbox)

| rank | mean_IoU | split/class | filename | (A,B,C) faceExp |
|---:|---:|---|---|---|
| 1 | 0.299 | train/anger | `8ogfa38652ea8c683e6e77ce6a635a586dec07484a7a8428ed5ba78e3d22cjubb.jpg` | 분노 / 분노 / 분노 |
| 2 | 0.302 | val/sadness | `ux6179a51e510953512d0c60796936550d4e2f94d929f1c415e900e095e95hl9z.jpg` | 슬픔 / 슬픔 / 슬픔 |
| 3 | 0.305 | train/sadness | `2iy7195d2defabe1b1761e8e148d1fe15d6960b8ddf87d74b78fbd3285da9u5x6.jpg` | 중립 / 불안 / 상처 |
| 4 | 0.314 | train/happy | `mbezd31b2d54ff4cc87290207fbe33ee4a7afe5344a788fe87cbce4fb8b1c4y2l.jpg` | 기쁨 / 기쁨 / 기쁨 |
| 5 | 0.328 | train/anger | `fper3db11fef98f236117e11df9f5ad4a93e1c9bf2172b1fa9bf41e47f2b17siz.jpg` | 분노 / 분노 / 분노 |
| 6 | 0.333 | train/happy | `k3s935e50e666fee414ce63c6a7cc0563dbee792014b083a74cbcbf602b735rqt.jpg` | 기쁨 / 기쁨 / 기쁨 |
| 7 | 0.576 | val/panic | `5u7u995b6cf179cc54e768ef75255576d22747468cd2fc94ff31cbe2fc677bmok.jpg` | 당황 / 당황 / 당황 |
| 8 | 0.646 | train/happy | `mcjld42681ab084d7243371ac8345626dc48d6b2615aa8804b573cbc74e4fw7x2.jpg` | 기쁨 / 기쁨 / 기쁨 |
| 9 | 0.671 | train/happy | `b0ex8c6c0edf5ea66a1dc61e1e96e640d61fef7cebca7d39346764c1e4a17b92q.jpg` | 기쁨 / 기쁨 / 기쁨 |
| 10 | 0.684 | train/sadness | `yzvsddac4a0d3d5fca4399b4cb5c9174204b560155f724cbd520e271422d4z2mb.jpg` | 슬픔 / 슬픔 / 슬픔 |

### 그리드 시각화에 포함된 샘플 (10장)

| # | split/class | filename | A / B / C |
|---:|---|---|---|
| 1 | train/anger | `b1cbe34734870cc11c33334e02bea93ac3a3b061caab62c0df1c6b9c75430tquz.jpg` | 상처 / 불안 / 슬픔 |
| 2 | train/anger | `eh1706d34df5aa3bc8ad927d82818b0e808e5d76d5f06c8f20adcb38f6f328m5t.jpg` | 슬픔 / 분노 / 불안 |
| 3 | train/anger | `az5sf92da0667e638d80d35ebce0238bb2c3c4a6cca44e153d08e0351aa6enbj5.jpg` | 상처 / 슬픔 / 분노 |
| 4 | train/anger | `6iamb8385d1913611ababcc9896fcae139ce5bb3acbef6f4afdd1ba89935e8udw.jpg` | 중립 / 불안 / 분노 |
| 5 | train/anger | `wguybd184c376887e68f99e67d4867bfcce1fc6d58852ce1a1c9c8e9cc67fsaol.jpg` | 불안 / 중립 / 당황 |
| 6 | train/anger | `oxbea3b555037fbe5c395b3bb96c5a84c34db5758fb52563c3efb2bfe3310y9ga.jpg` | 불안 / 분노 / 슬픔 |
| 7 | train/anger | `o4jo02bf1712cd8bb73bc0058e67c671bcdc2c1680ac462bca1f128d5cc6c60oi.jpg` | 불안 / 중립 / 분노 |
| 8 | train/anger | `cmjs059494a32a1f3827181a31a237eaaaf847b303369727444a421e194251xjz.jpg` | 상처 / 중립 / 분노 |
| 9 | train/anger | `qoz97bcd9b672e9b056664b2707f769273a7d7027e2e757c4c2de6be5ee16l18h.jpg` | 상처 / 슬픔 / 분노 |
| 10 | train/anger | `e5v08e375c9c8b57a484853d3988e84b4eb853deda53c3df751611346b4edgzzp.jpg` | 슬픔 / 불안 / 분노 |

## 5) 데이터 품질 인사이트

- 3인 bbox IoU 평균 0.920/0.936/0.925. 0.9 이상이면 라벨 일관성 우수 → annot_A bbox 단일 사용 정당화 가능. 0.7 미만이면 3명 평균 박스(median or union) 전처리 고려.
- 전원 일치율 61.1%, 완전 불일치 9.5%. 불일치가 큰 클래스는 모델 confusion 과도 직결 — loss weighting 또는 soft label 실험 후보.
- pairwise confusion 최다쌍 = 슬픔↔상처(729), 불안↔분노(665), 당황↔불안(580). 해당 쌍 간 구분이 모델에게도 어렵다는 신호 — feature aug(face crop, color jitter)로 보완.

## 6) 산출물
- `results/annot_consistency.md` (이 파일)
- `results/annot_disagreement_grid.png` — 논쟁 샘플 그리드
- `results/annot_iou_boxplot.png` — IoU 박스플롯