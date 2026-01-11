# -*- coding: utf-8 -*-
"""
CAD 图像网格分割模块（投影 + 空白带）（目前存在缺陷只能识别部分）

功能概述
1) 将 CAD 风格线稿图预处理为稳定的前景 mask（0/1）。
2) 基于投影（Y、X）检测空白带并做“多段切分”（一次切出多段），实现网格级区域划分：
   - Stage1：先按 Y 切出 rows，再按 X（可选）切出 cols/cells。
3) 可选二次细分（Refine）：仅对“细长且前景密度高”的区域继续按 Y 切分，解决漏切。
4) 输出调试框图、裁剪结果，并打印区域统计信息：
   - 位置百分比坐标采用“底部=0%，顶部=100%”（自下而上），修复你反馈的 y 方向反了问题。

使用方式
- 直接运行：修改 main() 里的 input_path。
- 或调用接口函数 partition_image_by_projection(...)，返回分割结果与 debug 信息。

如img.png/img_1.png/img_2.png/img_3.png可以进行划分
img_2.png/img_3.png 这种类型划分成功的概率不是很高，得看密度情况和是否是横平竖直，即规整度
img_4.png 由于很多标注存在重叠情况，所以无法进行划分
img_5.png 存在倾斜的图，使用图框则存在交叉现象，因此无法进行划分
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Rect:
    """图像矩形区域，半开区间 [x0,x1), [y0,y1)。"""
    x0: int
    y0: int
    x1: int
    y1: int
    fg_mass: float = 0.0
    tag: str = ""
    depth: int = 0

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class SplitConfig:
    """
    分割配置参数（集中管理，便于调参/复用）。

    Notes:
        - stage1：网格级分割（Y -> X）
        - refine：二次细分（仅对细长且密度高块做 Y）
    """
    # preprocess
    canny1: int = 50
    canny2: int = 150
    close_k_ratio: float = 0.006
    dilate_k_ratio: float = 0.003

    # projection common
    smooth_ratio: float = 0.02
    trim_ratio: float = 0.06
    gap_score_max: float = 0.50
    min_gap_width_ratio: float = 0.018
    merge_gap_ratio: float = 0.012

    # stage1: Y split
    y_q_blank: float = 0.12
    y_min_band_ratio: float = 0.012
    y_min_part_ratio: float = 0.08

    # stage1: X split
    x_q_blank: float = 0.12
    x_min_band_ratio: float = 0.012
    x_min_part_ratio: float = 0.20

    # stage1: conditional split-x
    only_split_x_for_dense_rows: bool = True
    dense_row_fg_ratio: float = 0.06

    # refine
    refine_rounds: int = 1
    refine_max_depth: int = 3
    refine_tall_ratio: float = 1.60
    refine_fg_density_min: float = 0.015

    # refine params (stricter)
    refine_smooth_ratio: float = 0.025
    refine_trim_ratio: float = 0.06
    refine_gap_score_max: float = 0.42
    refine_min_gap_width_ratio: float = 0.030
    refine_merge_gap_ratio: float = 0.015
    refine_y_q_blank: float = 0.10
    refine_y_min_band_ratio: float = 0.020
    refine_y_min_part_ratio: float = 0.28

    # outputs
    crop_pad_ratio: float = 0.01


# ============================================================
# 基础工具
# ============================================================

def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    """
    一维滑动平均平滑（自动转为奇数窗口）。

    Args:
        x: 1D 数组
        win: 窗口长度（>=1）

    Returns:
        平滑后的 float32 数组
    """
    win = int(max(1, win))
    if win % 2 == 0:
        win += 1
    if win == 1:
        return x.astype(np.float32)
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x.astype(np.float32), k, mode="same")


def preprocess_mask(img_bgr: np.ndarray, cfg: SplitConfig) -> np.ndarray:
    """
    将 CAD 彩色线稿图转为 0/1 前景 mask。

    Pipeline:
        gray -> blur -> Canny -> close -> dilate -> 0/1 mask

    Args:
        img_bgr: BGR 图像
        cfg: SplitConfig

    Returns:
        mask01: uint8，0/1
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, cfg.canny1, cfg.canny2)

    h, w = edges.shape
    k = max(3, int(round(min(h, w) * cfg.close_k_ratio)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    k2 = max(3, int(round(min(h, w) * cfg.dilate_k_ratio)))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
    dil = cv2.dilate(closed, kernel2, iterations=1)

    return (dil > 0).astype(np.uint8)


def region_fg_mass(mask01: np.ndarray, r: Rect) -> float:
    """计算 Rect 内前景像素总量。"""
    if r.w <= 0 or r.h <= 0:
        return 0.0
    return float(mask01[r.y0:r.y1, r.x0:r.x1].sum())


# ============================================================
# 投影分割核心（统一链路）
# ============================================================

def find_blank_bands(proj: np.ndarray, q_blank: float, min_band: int) -> List[Tuple[int, int]]:
    """
    低投影连续区间检测为空白带：
        thr = quantile(proj, q_blank)
        proj <= thr 视为 blank
        连续 blank 长度 >= min_band 记为 band

    Args:
        proj: 1D 投影（建议已平滑）
        q_blank: 分位数阈值
        min_band: 最小空白带长度

    Returns:
        [(b0,b1), ...] 半开区间
    """
    thr = np.quantile(proj, q_blank)
    is_blank = proj <= thr

    bands: List[Tuple[int, int]] = []
    i, n = 0, len(is_blank)
    while i < n:
        if not is_blank[i]:
            i += 1
            continue
        j = i
        while j < n and is_blank[j]:
            j += 1
        if (j - i) >= min_band:
            bands.append((i, j))
        i = j
    return bands


def filter_bands_by_gap_strength(proj_s: np.ndarray, bands: List[Tuple[int, int]], gap_score_max: float) -> List[Tuple[int, int]]:
    """
    过滤空白带强度：
        gap_score = mean(band) / mean(all)
        gap_score 越小越“空”

    Args:
        proj_s: 平滑投影
        bands: 空白带列表
        gap_score_max: 最大 gap_score

    Returns:
        过滤后的 bands
    """
    if not bands:
        return []
    all_mean = float(proj_s.mean()) + 1e-6
    kept: List[Tuple[int, int]] = []
    for b0, b1 in bands:
        band_mean = float(proj_s[b0:b1].mean()) if b1 > b0 else float(proj_s[b0])
        if (band_mean / all_mean) <= gap_score_max:
            kept.append((b0, b1))
    return kept


def merge_close_bands(bands: List[Tuple[int, int]], merge_gap: int) -> List[Tuple[int, int]]:
    """
    合并相邻/距离近的空白带，减少碎裂。

    Args:
        bands: 空白带列表
        merge_gap: 若 b0 <= prev_end + merge_gap，则合并

    Returns:
        合并后的 bands
    """
    if not bands:
        return []
    bands = sorted(bands, key=lambda x: x[0])
    merged = [bands[0]]
    for b0, b1 in bands[1:]:
        p0, p1 = merged[-1]
        if b0 <= p1 + merge_gap:
            merged[-1] = (p0, max(p1, b1))
        else:
            merged.append((b0, b1))
    return merged


def compute_projection(mask01: np.ndarray, r: Rect, axis: str, smooth_ratio: float, trim_ratio: float) -> Tuple[np.ndarray, int]:
    """
    对 Rect 做投影并平滑，同时对“另一维”裁边抑制边框/高亮干扰。

    Args:
        mask01: 0/1 mask
        r: Rect
        axis: "y" 或 "x"
        smooth_ratio: 平滑窗口比例（相对轴向长度）
        trim_ratio: 裁边比例（相对另一维长度）

    Returns:
        proj_s: 平滑投影
        L: 轴向长度（proj_s 长度）
    """
    sub = mask01[r.y0:r.y1, r.x0:r.x1]
    h, w = sub.shape
    if h <= 0 or w <= 0:
        return np.zeros((1,), np.float32), 0

    if axis == "y":
        pad = int(round(w * trim_ratio))
        sub2 = sub[:, pad:w - pad] if (w - 2 * pad) >= 50 else sub
        proj = sub2.sum(axis=1).astype(np.float32)
    elif axis == "x":
        pad = int(round(h * trim_ratio))
        sub2 = sub[pad:h - pad, :] if (h - 2 * pad) >= 50 else sub
        proj = sub2.sum(axis=0).astype(np.float32)
    else:
        raise ValueError("axis must be 'y' or 'x'")

    L = int(proj.shape[0])
    win = max(3, int(round(max(1, L) * smooth_ratio)))
    return moving_average_1d(proj, win), L


def split_rect_multi(
    mask01: np.ndarray,
    r: Rect,
    axis: str,
    *,
    q_blank: float,
    min_band_ratio: float,
    min_part_ratio: float,
    smooth_ratio: float,
    trim_ratio: float,
    gap_score_max: float,
    min_gap_width_ratio: float,
    merge_gap_ratio: float,
) -> List[Rect]:
    """
    沿 axis 对 Rect 做一次“多段投影切分”（一次可切出多个子段）。

    约束（减碎裂关键）：
      - 空白带宽度 >= min_gap_width_ratio * L
      - 空白带足够空：gap_score <= gap_score_max
      - 子块长度 >= min_part_ratio * L
      - 空白带之间合并：merge_gap_ratio * L

    Returns:
        子 Rect 列表；若不可切分，返回 [r]
    """
    proj_s, L = compute_projection(mask01, r, axis, smooth_ratio, trim_ratio)
    if L <= 0:
        return [r]

    min_band = max(5, int(round(L * min_band_ratio)))
    bands = find_blank_bands(proj_s, q_blank, min_band)

    # (1) 过滤过窄 band
    min_gap_w = max(5, int(round(L * min_gap_width_ratio)))
    bands = [(b0, b1) for (b0, b1) in bands if (b1 - b0) >= min_gap_w]

    # (2) 过滤不够空
    bands = filter_bands_by_gap_strength(proj_s, bands, gap_score_max)

    # (3) 合并相邻 band
    merge_gap = max(1, int(round(L * merge_gap_ratio)))
    bands = merge_close_bands(bands, merge_gap)

    if not bands:
        return [r]

    # 以 band 中点作为切点
    cuts = sorted({int(round((b0 + b1) / 2.0)) for (b0, b1) in bands})
    cuts = [max(0, min(L, c)) for c in cuts]

    min_part = max(10, int(round(L * min_part_ratio)))
    pts = [0] + cuts + [L]

    intervals: List[Tuple[int, int]] = []
    for i in range(len(pts) - 1):
        s, e = pts[i], pts[i + 1]
        if (e - s) >= min_part:
            intervals.append((s, e))

    if len(intervals) <= 1:
        return [r]

    out: List[Rect] = []
    for s, e in intervals:
        if axis == "y":
            rr = Rect(r.x0, r.y0 + s, r.x1, r.y0 + e, depth=r.depth + 1)
        else:
            rr = Rect(r.x0 + s, r.y0, r.x0 + e, r.y1, depth=r.depth + 1)
        rr.fg_mass = region_fg_mass(mask01, rr)
        out.append(rr)

    return out


# ============================================================
# Stage1：网格分割（Y -> X）
# ============================================================

def grid_partition(mask01: np.ndarray, root: Rect, cfg: SplitConfig) -> Tuple[List[Rect], List[Rect]]:
    """
    网格级分割：
      1) root 沿 Y 多段切 -> rows
      2) 每个 row（可选）沿 X 多段切 -> cells
    """
    total_fg = float(mask01.sum()) + 1e-6

    rows = split_rect_multi(
        mask01, root, "y",
        q_blank=cfg.y_q_blank,
        min_band_ratio=cfg.y_min_band_ratio,
        min_part_ratio=cfg.y_min_part_ratio,
        smooth_ratio=cfg.smooth_ratio,
        trim_ratio=cfg.trim_ratio,
        gap_score_max=cfg.gap_score_max,
        min_gap_width_ratio=cfg.min_gap_width_ratio,
        merge_gap_ratio=cfg.merge_gap_ratio,
    )
    for i, r in enumerate(rows):
        r.tag = f"row{i}"

    cells: List[Rect] = []
    for i, row in enumerate(rows):
        row_fg_ratio = row.fg_mass / total_fg
        do_split_x = True
        if cfg.only_split_x_for_dense_rows and (row_fg_ratio < cfg.dense_row_fg_ratio):
            do_split_x = False

        if do_split_x:
            cols = split_rect_multi(
                mask01, row, "x",
                q_blank=cfg.x_q_blank,
                min_band_ratio=cfg.x_min_band_ratio,
                min_part_ratio=cfg.x_min_part_ratio,
                smooth_ratio=cfg.smooth_ratio,
                trim_ratio=cfg.trim_ratio,
                gap_score_max=cfg.gap_score_max,
                min_gap_width_ratio=cfg.min_gap_width_ratio,
                merge_gap_ratio=cfg.merge_gap_ratio,
            )
            if len(cols) > 1:
                for j, c in enumerate(cols):
                    c.tag = f"row{i}_col{j}"
                cells.extend(cols)
            else:
                cells.append(row)
        else:
            cells.append(row)

    rows.sort(key=lambda rr: (rr.y0, rr.x0))
    cells.sort(key=lambda rr: (rr.y0, rr.x0))
    return rows, cells


# ============================================================
# Refine：二次细分（只对细长+密度高块做 Y）
# ============================================================

def refine_cells(mask01: np.ndarray, cells: List[Rect], cfg: SplitConfig) -> List[Rect]:
    """
    二次细分：
      - 仅对满足：
          (h/w >= cfg.refine_tall_ratio) 且 (fg_density >= cfg.refine_fg_density_min)
        的 cell 做 Y 方向多段切分
      - 迭代 cfg.refine_rounds 轮（建议 1）
    """
    cur = list(cells)

    for _ in range(cfg.refine_rounds):
        changed = False
        nxt: List[Rect] = []

        for c in cur:
            if c.depth >= cfg.refine_max_depth:
                nxt.append(c)
                continue

            if c.fg_mass <= 0:
                c.fg_mass = region_fg_mass(mask01, c)

            area = float(c.w * c.h) + 1e-6
            fg_density = c.fg_mass / area
            if fg_density < cfg.refine_fg_density_min:
                nxt.append(c)
                continue

            if c.w > 0 and (c.h / max(1, c.w)) >= cfg.refine_tall_ratio:
                parts = split_rect_multi(
                    mask01, c, "y",
                    q_blank=cfg.refine_y_q_blank,
                    min_band_ratio=cfg.refine_y_min_band_ratio,
                    min_part_ratio=cfg.refine_y_min_part_ratio,
                    smooth_ratio=cfg.refine_smooth_ratio,
                    trim_ratio=cfg.refine_trim_ratio,
                    gap_score_max=cfg.refine_gap_score_max,
                    min_gap_width_ratio=cfg.refine_min_gap_width_ratio,
                    merge_gap_ratio=cfg.refine_merge_gap_ratio,
                )
                if len(parts) > 1:
                    changed = True
                    for k, p in enumerate(parts):
                        p.tag = (c.tag + f"_y{k}") if c.tag else f"y{k}"
                    nxt.extend(parts)
                else:
                    nxt.append(c)
            else:
                nxt.append(c)

        cur = sorted(nxt, key=lambda r: (r.y0, r.x0))
        if not changed:
            break

    return cur


# ============================================================
# 输出：调试框图 / 裁剪 / 打印尺度（修复 y 百分比方向）
# ============================================================

def draw_rects_debug(img_bgr: np.ndarray, rects: List[Rect], out_path: str) -> None:
    """绘制矩形框到原图用于调试。"""
    vis = img_bgr.copy()
    for i, r in enumerate(rects):
        cv2.rectangle(vis, (r.x0, r.y0), (r.x1, r.y1), (0, 0, 255), 2)
        label = r.tag if r.tag else str(i)
        cv2.putText(
            vis, label, (r.x0, max(0, r.y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )
    cv2.imwrite(out_path, vis)


def save_crops(img_bgr: np.ndarray, rects: List[Rect], out_dir: str, pad_ratio: float = 0.01) -> None:
    """裁剪并保存。"""
    os.makedirs(out_dir, exist_ok=True)
    H, W = img_bgr.shape[:2]
    for i, r in enumerate(rects):
        pad = int(round(min(r.w, r.h) * pad_ratio))
        x0 = max(0, r.x0 - pad)
        y0 = max(0, r.y0 - pad)
        x1 = min(W, r.x1 + pad)
        y1 = min(H, r.y1 + pad)
        crop = img_bgr[y0:y1, x0:x1]
        tag = r.tag if r.tag else f"cell{i}"
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_{tag}_x{x0}-{x1}_y{y0}-{y1}.png"), crop)


def print_rects(title: str, rects: List[Rect], img_w: int, img_h: int, total_fg: Optional[float] = None) -> None:
    """
    输出像素尺度 + 百分比尺度 + 前景占比/密度。

    关键修复：
      - y 方向百分比采用“底部=0%，顶部=100%”（自下而上）
      - 因为 OpenCV 图像坐标 y 向下增大，所以需要做转换：
          y_pct_from_bottom = (H - y) / H
    """
    total_area = float(img_w * img_h) + 1e-6
    total_fg = float(total_fg) if total_fg is not None else 0.0

    print(f"==== {title} ====")
    print(f"数量: {len(rects)}")

    for i, r in enumerate(rects):
        w_px, h_px = r.w, r.h
        w_pct = 100.0 * (w_px / max(1, img_w))
        h_pct = 100.0 * (h_px / max(1, img_h))
        area_pct = 100.0 * ((w_px * h_px) / total_area)

        x0_pct = 100.0 * (r.x0 / max(1, img_w))
        x1_pct = 100.0 * (r.x1 / max(1, img_w))

        # y 方向：底部=0%，顶部=100%
        y0_pct = 100.0 * ((img_h - r.y0) / max(1, img_h))
        y1_pct = 100.0 * ((img_h - r.y1) / max(1, img_h))
        y_lo, y_hi = (min(y0_pct, y1_pct), max(y0_pct, y1_pct))

        fg = float(r.fg_mass)
        fg_density = fg / (float(w_px * h_px) + 1e-6)
        fg_density_pct = 100.0 * fg_density

        fg_ratio_pct = 0.0
        if total_fg > 1e-6:
            fg_ratio_pct = 100.0 * (fg / total_fg)

        tag = r.tag or "-"
        print(
            f"  {i}: tag={tag} "
            f"x=[{r.x0},{r.x1}) y=[{r.y0},{r.y1}) "
            f"w={w_px}({w_pct:.1f}%) h={h_px}({h_pct:.1f}%) area={area_pct:.1f}% "
            f"pos(x:{x0_pct:.1f}%→{x1_pct:.1f}%, y:{y_lo:.1f}%→{y_hi:.1f}%) "
            f"fg_mass={fg:.0f} fg_ratio={fg_ratio_pct:.1f}% fg_density={fg_density_pct:.2f}% "
            f"depth={r.depth}"
        )


# ============================================================
# 对外接口函数
# ============================================================

def partition_image_by_projection(
    img_bgr: np.ndarray,
    *,
    cfg: Optional[SplitConfig] = None,
    out_dir: Optional[str] = None,
    debug_prefix: str = "debug",
    save_debug_images: bool = True,
    save_cropped_regions: bool = True,
) -> Dict[str, any]:
    """
    对输入图像进行投影网格分割，并可选输出调试图与裁剪结果。

    Args:
        img_bgr: 输入 BGR 图像（cv2.imread(..., cv2.IMREAD_COLOR)）
        cfg: SplitConfig（不传则使用默认）
        out_dir:
            输出目录（若为 None，则不落盘；若提供则写入 debug 图片与裁剪结果）
        debug_prefix:
            输出文件名前缀，例如 debug_cells_stage1.png 会变成 {debug_prefix}_cells_stage1.png
        save_debug_images: 是否保存 debug 框图
        save_cropped_regions: 是否保存裁剪图

    Returns:
        dict:
          {
            "mask01": np.ndarray,
            "rows": List[Rect],
            "cells_stage1": List[Rect],
            "cells_refined": List[Rect],
            "image_size": {"H": int, "W": int},
            "total_fg": float,
            "outputs": { "stage1_boxes": path, "refined_boxes": path, "crops_dir": dir }  # 可能为空
          }
    """
    if cfg is None:
        cfg = SplitConfig()

    mask01 = preprocess_mask(img_bgr, cfg)
    total_fg = float(mask01.sum()) + 1e-6
    H, W = mask01.shape

    root = Rect(0, 0, W, H, fg_mass=float(mask01.sum()), tag="root", depth=0)

    # Stage1：网格分割
    rows, cells = grid_partition(mask01, root, cfg)

    # Refine：二次细分
    refined = refine_cells(mask01, cells, cfg)

    outputs: Dict[str, str] = {}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        if save_debug_images:
            p1 = os.path.join(out_dir, f"{debug_prefix}_cells_stage1.png")
            p2 = os.path.join(out_dir, f"{debug_prefix}_cells_refined.png")
            draw_rects_debug(img_bgr, cells, p1)
            draw_rects_debug(img_bgr, refined, p2)
            outputs["stage1_boxes"] = p1
            outputs["refined_boxes"] = p2

        if save_cropped_regions:
            crops_dir = os.path.join(out_dir, f"{debug_prefix}_crops")
            save_crops(img_bgr, refined, crops_dir, pad_ratio=cfg.crop_pad_ratio)
            outputs["crops_dir"] = crops_dir

    return {
        "mask01": mask01,
        "rows": rows,
        "cells_stage1": cells,
        "cells_refined": refined,
        "image_size": {"H": H, "W": W},
        "total_fg": total_fg,
        "outputs": outputs,
    }


# ============================================================
# main（示例）
# ============================================================

# def main():
#     input_path = "img.png"
#     out_dir = "out_grid_refined"
#
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"找不到输入文件：{input_path}")
#
#     img = cv2.imread(input_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise RuntimeError(f"无法读取图片：{input_path}")
#
#     cfg = SplitConfig()  # 需要调参就改这里（或改 SplitConfig 默认值）
#
#     res = partition_image_by_projection(
#         img,
#         cfg=cfg,
#         out_dir=out_dir,
#         debug_prefix="demo",
#         save_debug_images=True,
#         save_cropped_regions=True,
#     )
#
#     H = res["image_size"]["H"]
#     W = res["image_size"]["W"]
#     total_fg = res["total_fg"]
#
#     print_rects("Y 分区（rows）", res["rows"], img_w=W, img_h=H, total_fg=total_fg)
#     print_rects("初始区域（cells）", res["cells_stage1"], img_w=W, img_h=H, total_fg=total_fg)
#     print_rects("细分后区域（refined）", res["cells_refined"], img_w=W, img_h=H, total_fg=total_fg)
#
#     print(f"[DONE] outputs: {res['outputs']}")
#
#
# if __name__ == "__main__":
#     main()
