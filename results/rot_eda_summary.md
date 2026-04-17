# EDA 요약 — 감정 분류 프로젝트

## 전체 수량
- 이미지 총: **7196** (train 5996 / val 1200)
- 깨진/열기 실패: 0 (샘플링 기준)

## 클래스 분포

| 클래스 | train | val | 합계 |
|---|---:|---:|---:|
| anger (화남) | 1500 | 300 | 1800 |
| happy (행복) | 1495 | 300 | 1795 |
| panic (놀람/공포) | 1501 | 300 | 1801 |
| sadness (슬픔) | 1500 | 300 | 1800 |

## 이미지 해상도 (샘플링)
- 너비  W : min 1330, median 3088, max 9248
- 높이  H : min 1000, median 2208, max 6936
- 종횡비 W/H: median 1.33  (0.75 ~ 2.22)
- 224×224 resize 대상: 원본 대체로 고해상도 → 다운샘플링 손실 주의

## 이미지 채널 모드 (PIL `Image.mode`)
- `RGB` : 2400

## 이미지 ↔ 라벨 매칭

| split/class | 이미지 | 라벨 | 매칭 | img전용 | lbl전용 |
|---|---:|---:|---:|---:|---:|
| train/anger | 1500 | 1500 | 1500 | 0 | 0 |
| train/happy | 1495 | 1494 | 1494 | 1 | 0 |
| train/panic | 1501 | 1500 | 1500 | 1 | 0 |
| train/sadness | 1500 | 1500 | 1500 | 0 | 0 |
| val/anger | 300 | 300 | 300 | 0 | 0 |
| val/happy | 300 | 300 | 300 | 0 | 0 |
| val/panic | 300 | 300 | 300 | 0 | 0 |
| val/sadness | 300 | 300 | 300 | 0 | 0 |

- 매칭 불일치 총: **2**

## 라벨 부가 정보 (train)
- 연령 n=5994: mean 26.4, median 20, min 10, max 60
- 성별: 여=3236, 남=2758
- isProf: 일반인=3187, 전문인=2807

## Segmentation 요약

- train/anger: 1500 masks, sample shape=(3024, 4032), vals=[0, 1, 2, 3, 4]
- train/happy: 1495 masks, sample shape=(2320, 3088), vals=[0, 1, 2, 3, 4, 5]
- train/panic: 1501 masks, sample shape=(1737, 3088), vals=[0, 1, 2, 3, 4]
- train/sadness: 1500 masks, sample shape=(1932, 2576), vals=[0, 1, 2, 3, 4]
- val/anger: 300 masks, sample shape=(1528, 3216), vals=[0, 1, 2, 3, 4, 5]
- val/happy: 300 masks, sample shape=(1080, 1440), vals=[0, 1, 2, 3, 4, 5]
- val/panic: 300 masks, sample shape=(1592, 3264), vals=[0, 1, 2, 3, 4, 5]
- val/sadness: 300 masks, sample shape=(1240, 1654), vals=[0, 1, 2, 3, 4]

## 산출물
- `results/eda_report.png` — 분포 차트 6개
- `results/samples_grid.png` — 클래스별 샘플 + bbox 오버레이
- `results/seg_overlay.png` — segmentation 마스크 시각화
- `results/bbox_analysis.png` — bbox 크기/종횡비 박스플롯

## 판단 포인트
- 클래스 거의 균등 → class_weight 큰 조정 불필요
- 원본 해상도 편차 큼 → 224 resize 시 정보 손실 → face crop(annot_A) 필수
- seg 마스크로 배경/옷 제거 실험 → 창의성 + 모델링 점수 공략