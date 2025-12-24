"""PNG分块切图脚本
- 算法：二值化 → 连通域 → 可选投影分割
- 用途：把图中的有效内容块裁剪成独立PNG
"""

import os
from typing import List, Tuple

from PIL import Image, ImageDraw
import numpy as np


def to_binary(img: Image.Image, threshold: int = None, invert: bool = True) -> np.ndarray:
    """将图像转为灰度并按阈值二值化；threshold为空时自适应，invert可反相"""
    g = img.convert("L")
    a = np.asarray(g)

    if threshold is None:
        hist = np.bincount(a.reshape(-1), minlength=256)
        bg = np.argmin(hist)
        threshold = max(5, int((bg + 10)))
    b = (a > threshold)
    if invert:
        b = ~b
    binary_img = Image.fromarray((b * 255).astype(np.uint8), mode="L")
    binary_img.show() 
    return  





def connected_components_bbox(binary: np.ndarray, min_area: int = 5000) -> List[Tuple[int, int, int, int]]:
    """8邻域连通域提取，返回各块的外接矩形；min_area过滤小噪声"""
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=np.uint8)

    bboxes: List[Tuple[int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            minx, miny, maxx, maxy = x, y, x, y
            count = 0
            while stack:
                cy, cx = stack.pop()
                count += 1
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= h:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= w:
                            continue
                        if visited[ny, nx]:
                            continue
                        if not binary[ny, nx]:
                            continue
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            if count >= min_area:
                bboxes.append((minx, miny, maxx, maxy))
    return bboxes

def merge_close_bboxes(bboxes: List[Tuple[int, int, int, int]], min_distance: int) -> List[Tuple[int, int, int, int]]:
    """合并相近的边界框，如果它们之间的距离小于min_distance（支持迭代合并）"""
    if not bboxes or min_distance <= 0:
        return bboxes
    
    def merge_pass(bbox_list):
        bbox_list = sorted(bbox_list, key=lambda b: (b[1], b[0]))
        merged = []
        current = list(bbox_list[0])
        for next_box in bbox_list[1:]:
            cx0, cy0, cx1, cy1 = current
            nx0, ny0, nx1, ny1 = next_box
            dx = max(0, max(cx0 - nx1, nx0 - cx1))
            dy = max(0, max(cy0 - ny1, ny0 - cy1))
            dist = max(dx, dy)
            if dist < min_distance:
                current[0] = min(cx0, nx0)
                current[1] = min(cy0, ny0)
                current[2] = max(cx1, nx1)
                current[3] = max(cy1, ny1)
            else:
                merged.append(tuple(current))
                current = list(next_box)
        merged.append(tuple(current))
        return merged
    
    # 迭代合并直到稳定
    prev_len = len(bboxes)
    while True:
        bboxes = merge_pass(bboxes)
        if len(bboxes) == prev_len:
            break
        prev_len = len(bboxes)
    return bboxes


def merge_overlapping_bboxes(bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    if not bboxes:
        return bboxes
    arr = [list(b) for b in bboxes]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(arr)
        for i in range(len(arr)):
            if used[i]:
                continue
            cur = arr[i]
            for j in range(i + 1, len(arr)):
                if used[j]:
                    continue
                other = arr[j]
                if not (cur[2] < other[0] or other[2] < cur[0] or cur[3] < other[1] or other[3] < cur[1]):
                    cur = [min(cur[0], other[0]), min(cur[1], other[1]), max(cur[2], other[2]), max(cur[3], other[3])]
                    used[j] = True
                    changed = True
            out.append(cur)
            used[i] = True
        arr = out
    return [tuple(b) for b in arr]

def _segments_from_proj(mask: np.ndarray, gap_len: int) -> List[Tuple[int, int]]:
    """从一维投影布尔序列中提取连续段，允许空隙长度为gap_len"""
    segs: List[Tuple[int, int]] = []
    n = mask.shape[0]

    i = 0
    while i < n:
        while i < n and not mask[i]:
            i += 1
        if i >= n:
            break
        j = i
        gap = 0
        while j < n:
            if mask[j]:
                gap = 0
            else:
                gap += 1
                if gap >= gap_len:
                    break
            j += 1
        end = j - gap - 1
        segs.append((i, end))
        i = end + gap_len + 1
    return segs


def projection_split(binary: np.ndarray, min_gap: int = 10, min_area: int = 4000) -> List[Tuple[int, int, int, int]]:
    """基于水平/垂直投影对大块进行进一步分割"""
    h, w = binary.shape
    col_has = (binary.sum(axis=0) > 0)
    row_has = (binary.sum(axis=1) > 0)
    xs = _segments_from_proj(col_has, min_gap)
    ys = _segments_from_proj(row_has, min_gap)
    out: List[Tuple[int, int, int, int]] = []
    for x0, x1 in xs:
        for y0, y1 in ys:
            sub = binary[y0 : y1 + 1, x0 : x1 + 1]
            if int(sub.sum()) >= min_area:
                out.append((x0, y0, x1, y1))
    return out


def clip_and_save(img: Image.Image, bboxes: List[Tuple[int, int, int, int]], out_dir: str, prefix: str, pad: int = 8) -> List[str]:
    """按外接矩形裁剪并保存PNG；pad为留边像素"""
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []
    w, h = img.size
    for i, (x0, y0, x1, y1) in enumerate(bboxes, 1):
        x0p = max(0, x0 - pad)
        y0p = max(0, y0 - pad)
        x1p = min(w - 1, x1 + pad)
        y1p = min(h - 1, y1 + pad)
        crop = img.crop((x0p, y0p, x1p + 1, y1p + 1))
        out_path = os.path.join(out_dir, f"{prefix}_{i:03d}.png")
        crop.save(out_path)
        saved.append(out_path)
    return saved


def process_image(path: str, out_dir: str, threshold: int = None, extend: int = 50) -> List[str]:
    """先横向按投影分段（允许≤extend的空隙），边界再各延伸extend；再对每段纵向同样处理"""
    img = Image.open(path)
    bin0 = to_binary(img, threshold=threshold)
    h, w = bin0.shape
    row_has = (bin0.sum(axis=1) > 0)
    y_segs = _segments_from_proj(row_has, extend)
    bboxes: List[Tuple[int, int, int, int]] = []
    for y0, y1 in y_segs:
        y0e = max(0, y0 - extend)
        y1e = min(h - 1, y1 + extend)
        col_has = (bin0[y0e : y1e + 1, :].sum(axis=0) > 0)
        x_segs = _segments_from_proj(col_has, extend)
        for x0, x1 in x_segs:
            x0e = max(0, x0 - extend)
            x1e = min(w - 1, x1 + extend)
            bboxes.append((x0e, y0e, x1e, y1e))
    bboxes = merge_overlapping_bboxes(bboxes)
    saved = clip_and_save(img, bboxes, out_dir, prefix=os.path.splitext(os.path.basename(path))[0], pad=0)
    return saved


def demo(out_dir: str) -> List[str]:
    """生成一张示例图并进行分块，便于快速验证算法"""
    os.makedirs(out_dir, exist_ok=True)
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((60, 60, 520, 300), outline=(255, 255, 255), width=3)
    d.rectangle((80, 80, 240, 160), outline=(255, 255, 255), width=2)
    d.rectangle((300, 100, 500, 200), outline=(255, 255, 255), width=2)
    d.rectangle((900, 120, 1500, 500), outline=(255, 255, 255), width=3)
    d.rectangle((920, 140, 1100, 220), outline=(255, 255, 255), width=2)
    d.rectangle((1150, 140, 1470, 220), outline=(255, 255, 255), width=2)
    src = os.path.join(out_dir, "demo.png")
    img.save(src)
    return process_image(src, out_dir=os.path.join(out_dir, "clips"), threshold=20, extend=50)


def main():
    # 中文：通过下方变量控制脚本行为，无需命令行参数
    DEMO_MODE = False               # True 则运行内置示例并输出分块结果
    input_path = r"G:\Desktop\test\1.png"               # 要处理的PNG绝对路径；DEMO_MODE=False时必须填写
    out_dir = None                  # 输出目录；None时自动在图片同目录创建 *_clips
    threshold = None                # 二值化阈值；None为自适应
    extend = 50                     # 回退/延伸像素数
    
    if DEMO_MODE:
        demo_out = out_dir or os.path.join(os.path.dirname(__file__), "demo_out")
        saved = demo(demo_out)
    else:
        if not input_path:
            raise ValueError("请在主函数中设置 input_path 为待处理的PNG路径")
        out_dir = out_dir or os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0] + "_clips")
        saved = process_image(
            path=input_path,
            out_dir=out_dir,
            threshold=threshold,
            extend=extend,
        )
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()