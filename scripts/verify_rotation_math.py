"""EXIF orientation 회전 수학 독립 검증 스크립트.

normalize_orientation.py 의 rotate_bbox / rotate_mask_np / PIL_OP 를
수정 없이 import 해서, 각 orientation 1~8 에 대해
  (a) mask: PIL Image.transpose(PIL_OP[ori]) == rotate_mask_np
  (b) bbox: 네 코너 직접 회전 후 min/max == rotate_bbox
가 성립하는지 픽셀 단위 / 수학적 일치를 검증한다.

실행: python scripts/verify_rotation_math.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from normalize_orientation import PIL_OP, rotate_bbox, rotate_mask_np  # noqa: E402


# ────────────────────────────────────────────────────────────────
# 도우미: "연속 좌표" 컨벤션으로 코너를 직접 회전 시키는 ground truth.
#  (0,0)=왼쪽 위 모서리, (W,H)=오른쪽 아래 모서리. 픽셀 중심 아님.
#  PIL.Image.transpose 매핑:
#   2 FLIP_LEFT_RIGHT : (x,y) → (W-x, y)
#   3 ROTATE_180      : (x,y) → (W-x, H-y)
#   4 FLIP_TOP_BOTTOM : (x,y) → (x, H-y)
#   5 TRANSPOSE       : (x,y) → (y, x)                 new size=(H,W)
#   6 ROTATE_270(CW90): (x,y) → (H-y, x)               new size=(H,W)
#   7 TRANSVERSE      : (x,y) → (H-y, W-x)             new size=(H,W)
#   8 ROTATE_90 (CCW) : (x,y) → (y, W-x)               new size=(H,W)
# ────────────────────────────────────────────────────────────────
def corner_transform(x, y, ori, W, H):
    if ori == 1: return (x, y)
    if ori == 2: return (W - x, y)
    if ori == 3: return (W - x, H - y)
    if ori == 4: return (x, H - y)
    if ori == 5: return (y, x)
    if ori == 6: return (H - y, x)
    if ori == 7: return (H - y, W - x)
    if ori == 8: return (y, W - x)
    raise ValueError(ori)


def gt_bbox(box, ori, W, H):
    x0, x1 = box["minX"], box["maxX"]
    y0, y1 = box["minY"], box["maxY"]
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    tc = [corner_transform(x, y, ori, W, H) for x, y in corners]
    xs = [p[0] for p in tc]
    ys = [p[1] for p in tc]
    return {"minX": min(xs), "maxX": max(xs),
            "minY": min(ys), "maxY": max(ys)}


# ────────────────────────────────────────────────────────────────
# mask 검증
# ────────────────────────────────────────────────────────────────
def make_marker_mask(W, H):
    """각 픽셀값이 (y * W + x) 인 mask — 모든 픽셀이 유일."""
    m = np.arange(W * H, dtype=np.int32).reshape(H, W)
    return m


def pil_rotate_mask_gt(mask, ori):
    """PIL transpose 기반 ground truth. int32 → I 모드 이용."""
    if ori == 1:
        return mask.copy()
    # PIL 'I' 는 32-bit signed integer
    img = Image.fromarray(mask, mode="I")
    out = img.transpose(PIL_OP[ori])
    return np.array(out, dtype=np.int32)


def test_masks():
    results = []
    # 다양한 shape 테스트 (정사각, 가로 긴, 세로 긴, 홀수)
    shapes = [(4, 4), (3, 5), (5, 3), (7, 4), (1, 6), (6, 1), (1, 1), (8, 8)]
    for H, W in shapes:
        mask = make_marker_mask(W, H)
        for ori in range(1, 9):
            ours = rotate_mask_np(mask, ori)
            gt = pil_rotate_mask_gt(mask, ori)
            ok = (ours.shape == gt.shape) and np.array_equal(ours, gt)
            results.append((f"mask H={H} W={W} ori={ori}",
                            ok, ours.shape, gt.shape))
    return results


# ────────────────────────────────────────────────────────────────
# bbox 검증
# ────────────────────────────────────────────────────────────────
def approx_eq_box(a, b, tol=1e-9):
    return all(abs(a[k] - b[k]) <= tol for k in ("minX", "maxX", "minY", "maxY"))


def test_bboxes():
    results = []
    canvases = [(10, 10), (20, 8), (8, 20), (13, 7), (100, 100)]
    # (x0,y0,x1,y1) cases — 정상 / 코너 / 경계 / 1px / 0크기 / float
    def boxes_for(W, H):
        return [
            # 중앙 정사각
            (W*0.3, H*0.3, W*0.7, H*0.7),
            # 가로 긴
            (1, 1, W-1, 3),
            # 세로 긴
            (1, 1, 3, H-1),
            # 좌상 코너 닿음
            (0, 0, W*0.5, H*0.5),
            # 우하 코너 닿음
            (W*0.5, H*0.5, W, H),
            # 전체
            (0, 0, W, H),
            # 1픽셀
            (2, 3, 3, 4),
            # 크기 0 (점)
            (2.5, 3.5, 2.5, 3.5),
            # float 좌표
            (1.234, 2.567, 5.89, 6.123),
            # 오른쪽 경계 exactly
            (W-1, 0, W, H),
            # 아래 경계 exactly
            (0, H-1, W, H),
        ]
    for (W, H) in canvases:
        for (x0, y0, x1, y1) in boxes_for(W, H):
            box = {"minX": x0, "maxX": x1, "minY": y0, "maxY": y1}
            for ori in range(1, 9):
                ours = rotate_bbox(box, ori, W, H)
                gt = gt_bbox(box, ori, W, H)
                ok = approx_eq_box(ours, gt)
                results.append((
                    f"bbox W={W} H={H} box=({x0},{y0},{x1},{y1}) ori={ori}",
                    ok, ours, gt))
    return results


# ────────────────────────────────────────────────────────────────
# 추가: bbox 가 이미지 범위 밖을 벗어나지 않는지 (W-x 컨벤션이 올바른가)
# ────────────────────────────────────────────────────────────────
def test_bbox_in_bounds():
    """회전 후 bbox 는 new image (W',H') 안에 포함되어야 한다."""
    results = []
    for (W, H) in [(10, 10), (20, 8), (8, 20)]:
        box = {"minX": 0, "maxX": W, "minY": 0, "maxY": H}
        for ori in range(1, 9):
            new_W, new_H = (W, H) if ori in (1, 2, 3, 4) else (H, W)
            r = rotate_bbox(box, ori, W, H)
            ok = (0 <= r["minX"] <= r["maxX"] <= new_W and
                  0 <= r["minY"] <= r["maxY"] <= new_H)
            results.append((f"in-bounds W={W} H={H} ori={ori}",
                            ok, r, (new_W, new_H)))
    return results


# ────────────────────────────────────────────────────────────────
# 추가: mask 회전 후 shape 예상과 일치하는가
# ────────────────────────────────────────────────────────────────
def test_mask_shape():
    results = []
    H, W = 5, 9
    mask = make_marker_mask(W, H)
    for ori in range(1, 9):
        out = rotate_mask_np(mask, ori)
        exp = (H, W) if ori in (1, 2, 3, 4) else (W, H)
        ok = out.shape == exp
        results.append((f"mask-shape ori={ori}", ok, out.shape, exp))
    return results


# ────────────────────────────────────────────────────────────────
# 리포트
# ────────────────────────────────────────────────────────────────
def summarize(name, rows):
    fails = [r for r in rows if not r[1]]
    print(f"\n=== {name}: total={len(rows)}, "
          f"PASS={len(rows)-len(fails)}, FAIL={len(fails)} ===")
    if fails:
        for r in fails[:20]:
            print("  FAIL:", r)
    return len(fails) == 0


def main():
    ok1 = summarize("mask pixel-equality vs PIL",  test_masks())
    ok2 = summarize("bbox vs corner-rotation GT",  test_bboxes())
    ok3 = summarize("bbox in-bounds",              test_bbox_in_bounds())
    ok4 = summarize("mask output shape",           test_mask_shape())
    final = ok1 and ok2 and ok3 and ok4
    print("\n=========================================")
    print(" FINAL:", "GO (ALL PASS)" if final else "NO-GO (SEE FAILS ABOVE)")
    print("=========================================")
    sys.exit(0 if final else 1)


if __name__ == "__main__":
    main()
