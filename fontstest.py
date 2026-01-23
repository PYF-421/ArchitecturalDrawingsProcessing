"""
DXF图框转PNG工具

"""

import ezdxf
from ezdxf.colors import DXF_DEFAULT_COLORS
from PIL import Image, ImageDraw, ImageFont
import math
import re
import csv
import os
import gc
import re
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
# 大模型判断切分行列数（可选）
ENABLE_LLM_GRID = True  # 是否启用大模型视觉判断行列（影响 mask 生成流程）
LLM_MAX_COLS = 6  # 发送给大模型的最大列数
LLM_MAX_ROWS = 6  # 发送给大模型的最大行数
LLM_MASK_CANNY1 = 60  # mask 生成 Canny 的低阈值（影响 mask 细节）
LLM_MASK_CANNY2 = 180  # mask 生成 Canny 的高阈值（影响 mask 细节）
LLM_MASK_CLOSE_K_RATIO = 0.004  # mask 闭运算核宽占图像宽的比例（mask 生成）
LLM_MASK_DILATE_K_RATIO = 0.002  # mask 膨胀核宽占图像宽的比例（mask 生成）
MASK_PROJ_Q_BLANK_X = 0.05  # X 方向投影中判断“空白带”的分位数（mask 判定使用）
MASK_PROJ_Q_BLANK_Y = 0.05  # Y 方向投影中判断“空白带”的分位数（mask 判定使用）
MASK_PROJ_Q_LINE_X = 0.85  # X 方向投影中判断“线条带”的分位数（mask 判定使用）
MASK_PROJ_Q_LINE_Y = 0.85  # Y 方向投影中判断“线条带”的分位数（mask 判定使用）
MASK_PROJ_MIN_BAND_RATIO = 0.003  # 投影带宽必须占图像尺寸的最小比例（mask 判定）
MASK_PROJ_MIN_PART_RATIO = 0.06  # 切分片段必须占图像尺寸的最小比例（mask 投影使用）
MASK_PROJ_EDGE_MARGIN_RATIO = 0.01  # 切分线不能太靠近图像边缘占比（mask 判断）
MASK_LINE_KERNEL_X_RATIO = 0.08  # 默认提取水平长线的核宽比例（mask 判定过滤文字）
MASK_LINE_KERNEL_Y_RATIO = 0.08  # 默认提取垂直长线的核高比例（mask 判定过滤文字）
MASK_LINE_KERNEL_X_RATIO_STRUCT = 0.3  # 结构化 mask 提取水平长线的核宽比例（用于 `_infer_layout_from_mask`）
MASK_LINE_KERNEL_Y_RATIO_STRUCT = 0.3  # 结构化 mask 提取垂直长线的核高比例（用于 `_infer_layout_from_mask`）
STRUCT_VLINE_COVER_RATIO = 0.1  # 纵向强线需覆盖图像高度的比例（mask 判定以确定强线）
STRUCT_HLINE_COVER_RATIO = 0.1  # 横向强线需覆盖图像宽度的比例（mask 判定以确定强线）
MASK_GRID_INTERSECTION_MIN = 12  # grid 交点数判定结构化的最小值
MASK_GRID_INTERSECTION_RATIO = 0.00008  # grid 交点像素占比
STRUCT_SNAP_MAX_RATIO = 0.02  # 结构化模式切分线吸附到强线的最大距离占比

# 用于 mask 侧布局分类的线条/强线相关阈值
LAYOUT_LINE_RATIO_STRUCT = 0.05  # 线条像素占比高于此并有强线才可能是 STRUCTURED
LAYOUT_LINE_RATIO_TEXT_MAX = 0.08  # 没有强线且线比低于此就判 TEXT_ONLY
LAYOUT_STRONG_V_MIN = 2  # 判定强纵线时至少需 2 条（辅助 mask 判定）
LAYOUT_STRONG_H_MIN = 1  # 判定强横线时至少需 1 条（辅助 mask 判定）
LAYOUT_STRUCT_SCORE_THRESHOLD = 2  # 结构评分的基础值（未直接使用，可保留）
LAYOUT_STRUCT_STRONG_H_MIN = 1  # 判 STRUCTURED 时需要的最少强横线数量
LAYOUT_STRUCT_STRONG_V_MIN = 1  # 判 STRUCTURED 时需要的最少强纵线数量
LAYOUT_STRUCT_AREA_RATIO = 0.10  # 强线面积占比需超过此才能认为是“覆盖大面积”
LAYOUT_STRUCT_CONFIDENCE_MIN = 0.30  # 结构化置信度最低值（各种因素加权后）
MASK_CONFIDENCE_THRESHOLD = 0.35  # mask 判定覆盖 LLM 的置信度阈值
MASK_CONFIDENCE_MIN_OVERRIDE = 0.2  # 置信度稍高时也可以让 mask 结果生效
from dataclasses import dataclass
from collections import defaultdict

from llm_call_tool import LlmCaller
from base64_util import ImageToBase64Converter
from process_cad_mask import preprocess_mask

# ============================================================================
# 参数配置
# ============================================================================
# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DXF_PATH = os.path.join(SCRIPT_DIR, "www.alltoall.net_-6审-住宅设计说明_t3_t3_XsuajLVCMT.dxf")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "fontstest-12")
# INPUT_DXF_PATH = r"interest.dxf"
# OUTPUT_DIR = r"fontstest"
# 背景颜色
BACKGROUND_COLOR = (33, 40, 48)

# 文字压缩参数
CHINESE_COMPRESSION = 0.8
TEXT_HORIZONTAL_SCALE = 0.8

# 防重叠开关
ENABLE_OVERLAP_PREVENTION = False

# 字体大小比例
CHINESE_FONT_SCALE = 0.8
WESTERN_FONT_SCALE = 0.9

# 输出设置
TARGET_DPI = 150
OUTPUT_DPI = 300
MAX_IMAGE_DIMENSION = 12000

# 切分设置
TARGET_COLS = 3                # 目标列数
TARGET_ROWS = 3                # 每列目标行数
CUT_OFFSET_X = 200             # 垂直切分点左移距离（DXF单位，mm）
CUT_OFFSET_Y = -200             # 水平切分点上移距离（DXF单位，mm）
ENABLE_SPLIT = True          # 是否启用切分功能
DEBUG_DRAW_LINES = True       # 是否绘制调试线条
STRUCTURED_FIXED_COLS = 3
STRUCTURED_FIXED_ROWS = 3
FORCE_STRUCTURED_FRAMES = set()
FORCE_TEXT_ONLY_FRAMES = set()

# 大块表格检测配置
ENABLE_TABLE_MODE = False       # 是否启用表格模式切分（False则只用文字聚类模式）
TABLE_SIZE_THRESHOLD = 0.7     # 大块表格尺寸阈值（占图框宽或高的比例）
MANUAL_CUT_X = None            # 手动指定纵向切分点，例如: [32700, 56700, 82500]
MANUAL_CUT_Y = None            # 手动指定横向切分点

# ============================================================================
# CAD特殊符号替换表
# ============================================================================
CAD_SYMBOL_TABLE = {
    '%%P': '±', '%%p': '±',
    '%%D': '°', '%%d': '°',
    '%%C': 'Ø', '%%c': 'Ø',
    '%%U': '', '%%u': '',
    '%%O': '', '%%o': '',
    '%%%': '%',
}


@dataclass
class FrameInfo:
    index: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    insert_x: float
    insert_y: float
    scale_x: float
    scale_y: float


@dataclass
class EntityBBox:
    """实体边界框（DXF世界坐标）"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    entity_type: str


class DXFConverter:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"加载DXF: {INPUT_DXF_PATH}")
        self.doc = ezdxf.readfile(INPUT_DXF_PATH)
        self.msp = self.doc.modelspace()
        
        self.layer_colors = {}
        for layer in self.doc.layers:
            color_index = layer.dxf.color
            self.layer_colors[layer.dxf.name] = self._aci_to_rgb(color_index)
        
        self._load_fonts()
        self._font_cache = {}
        self.frame_block_name = '民用院-A1H'
    
    def _aci_to_rgb(self, color_index: int) -> Tuple[int, int, int]:
        if color_index <= 0 or color_index >= 256:
            return (255, 255, 255)
        try:
            color_int = DXF_DEFAULT_COLORS[color_index]
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            return (r, g, b)
        except:
            return (255, 255, 255)
    
    def _load_fonts(self):
        for path in ["华文仿宋.TTF",
                     "华文仿宋.TTF",]:
            try:
                ImageFont.truetype(path, 20)
                self.chinese_font = path
                print(f"  中文字体: {path}")
                break
            except:
                continue
        else:
            self.chinese_font = None
        
        for path in ["等线.ttf",
                     "等线.ttf"]:

            try:
                ImageFont.truetype(path, 20)
                self.western_font = path
                print(f"  西文字体: {path}")
                break
            except:
                continue
        else:
            self.western_font = None

    def _infer_grid_from_llm(self, image_path: str) -> Optional[Tuple[int, int, str]]:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            return None
        api_url = os.environ.get("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        model_name = os.environ.get("DASHSCOPE_MODEL", "qwen-vl-max").strip()
        caller = LlmCaller(api_key=api_key, url=api_url, model_name=model_name)

        img_b64 = ImageToBase64Converter.image_to_base64(image_path, with_data_uri=False)
        prompt = (
            "你看到的是一张建筑图框的二值化mask图（白色为线/字，黑色为背景）。\n\n"
            "任务1：判别布局类型（非常重要）：\n"
            "- 只有当“规则、密集、长距离贯穿”的横竖线网格覆盖画面主要区域（>35%）时，才判为 type=STRUCTURED。\n"
            "- 如果只是局部小表格或少量框线，仍判为 type=TEXT_ONLY。\n"
            "- 如果线条主要是不规则短线/断裂文本行，没有大范围规则网格，判为 type=TEXT_ONLY。\n\n"
            "任务2：判断主区域分区的列数和行数（用于粗切图）。\n"
            "重要要求：\n"
            "1) 只看大的分区边界，忽略表格内部细小格子线。\n"
            f"2) 列数行数范围在 1~{LLM_MAX_COLS} / 1~{LLM_MAX_ROWS}。\n"
            "3) 如果不确定，优先输出更少的行列数。\n\n"
            "输出格式必须严格为：\n"
            "type=XXX, cols=X, rows=Y"
        )
        resp = caller.call(prompt, images_base64=[img_b64], temperature=0.2)
        if not resp:
            return None
        m = re.search(r"type\s*=\s*(\w+)\s*,\s*cols\s*=\s*([1-9][0-9]?)\s*,\s*rows\s*=\s*([1-9][0-9]?)", resp)
        if m:
            ltype = m.group(1).upper()
            cols = int(m.group(2))
            rows = int(m.group(3))
        else:
            m2 = re.search(r"cols\s*=\s*([1-9][0-9]?)\s*,\s*rows\s*=\s*([1-9][0-9]?)", resp)
            if not m2:
                return None
            ltype = "UNKNOWN"
            cols = int(m2.group(1))
            rows = int(m2.group(2))
        cols = max(1, min(LLM_MAX_COLS, cols))
        rows = max(1, min(LLM_MAX_ROWS, rows))
        if ltype == "UNKNOWN":
            print(f"    LLM类型未解析，回退为 UNKNOWN，原始回复: {resp}")
        return cols, rows, ltype

    def _build_llm_mask(self, full_png_path: str, frame_dir: str) -> Optional[str]:
        try:
            img_bgr = cv2.imread(full_png_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None
            mask01 = preprocess_mask(
                img_bgr,
                canny1=LLM_MASK_CANNY1,
                canny2=LLM_MASK_CANNY2,
                close_k_ratio=LLM_MASK_CLOSE_K_RATIO,
                dilate_k_ratio=LLM_MASK_DILATE_K_RATIO,
            )
            mask_path = os.path.join(frame_dir, "frame_mask01.png")
            cv2.imwrite(mask_path, mask01 * 255)
            return mask_path
        except Exception:
            return None

    def _find_blank_bands(self, proj: np.ndarray, q_blank: float, min_band_px: int):
        thr = np.quantile(proj, q_blank)
        is_blank = proj <= thr
        bands = []
        i = 0
        n = len(is_blank)
        while i < n:
            if not is_blank[i]:
                i += 1
                continue
            j = i
            while j < n and is_blank[j]:
                j += 1
            if (j - i) >= min_band_px:
                bands.append((i, j))
            i = j
        return bands

    def _extract_long_lines(self, mask01: np.ndarray,
                            kx_ratio: Optional[float] = None,
                            ky_ratio: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask01.shape
        kx = max(15, int(round(w * (kx_ratio if kx_ratio is not None else MASK_LINE_KERNEL_X_RATIO))))
        ky = max(15, int(round(h * (ky_ratio if ky_ratio is not None else MASK_LINE_KERNEL_Y_RATIO))))
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
        h_lines = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, kernel_h, iterations=1)
        v_lines = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, kernel_v, iterations=1)
        return h_lines, v_lines

    def _infer_layout_from_mask(self, mask_path: str) -> Tuple[str, float]:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return "STRUCTURED", 1.0
        mask01 = (mask > 0).astype(np.uint8) * 255
        h, w = mask01.shape
        h_lines, v_lines = self._extract_long_lines(
            mask01,
            kx_ratio=MASK_LINE_KERNEL_X_RATIO_STRUCT,
            ky_ratio=MASK_LINE_KERNEL_Y_RATIO_STRUCT
        )
        line_pixels = int((h_lines > 0).sum() + (v_lines > 0).sum())
        mask_pixels = int((mask01 > 0).sum())
        line_ratio = line_pixels / max(1, mask_pixels)

        min_band_x = max(3, int(round(w * MASK_PROJ_MIN_BAND_RATIO)))
        min_band_y = max(3, int(round(h * MASK_PROJ_MIN_BAND_RATIO)))
        v_strength = (v_lines > 0).sum(axis=0)
        h_strength = (h_lines > 0).sum(axis=1)
        min_v_cover = int(round(h * STRUCT_VLINE_COVER_RATIO))
        min_h_cover = int(round(w * STRUCT_HLINE_COVER_RATIO))

        strong_v = self._find_strong_line_bands(v_strength, min_v_cover, min_band_x)
        strong_h = self._find_strong_line_bands(h_strength, min_h_cover, min_band_y)

        strong_mask = cv2.bitwise_or(h_lines, v_lines)
        strong_area_pixels = int((strong_mask > 0).sum())
        strong_area_ratio = strong_area_pixels / max(1, h * w)
        intersections = cv2.bitwise_and(h_lines, v_lines)
        num_grid_nodes, intersection_pixels = self._count_grid_intersections(intersections)
        intersection_ratio = intersection_pixels / max(1, h * w)

        has_sufficient_lines = (
            len(strong_h) >= LAYOUT_STRUCT_STRONG_H_MIN
            and len(strong_v) >= LAYOUT_STRUCT_STRONG_V_MIN
        )

        struct_strength = strong_area_ratio * 0.6 + line_ratio * 0.4
        confidence = min(1.0, struct_strength + 0.1)
        print(
            f"    Mask布局统计: strong_h={len(strong_h)}, strong_v={len(strong_v)}, "
            f"strong_area_ratio={strong_area_ratio:.3f}, line_ratio={line_ratio:.3f}, "
            f"struct_strength={struct_strength:.3f}, has_sufficient_lines={has_sufficient_lines}, "
            f"grid_nodes={num_grid_nodes}, grid_ratio={intersection_ratio:.5f}"
        )
        if (
            has_sufficient_lines
            and struct_strength >= LAYOUT_STRUCT_CONFIDENCE_MIN
            and strong_area_ratio >= LAYOUT_STRUCT_AREA_RATIO
        ):
            return "STRUCTURED", confidence

        grid_trigger = (
            num_grid_nodes >= MASK_GRID_INTERSECTION_MIN
            and intersection_ratio >= MASK_GRID_INTERSECTION_RATIO
            and line_ratio >= 0.25
        )
        if grid_trigger:
            print("    Mask网格交点触发结构化策略")
            return "STRUCTURED", min(1.0, confidence + 0.1)

        if len(strong_v) == 0 and len(strong_h) == 0 and line_ratio <= LAYOUT_LINE_RATIO_TEXT_MAX:
            return "TEXT_ONLY", max(0.2, 1 - strong_area_ratio)

        return "TEXT_ONLY", max(0.1, min(0.9, 1 - strong_area_ratio))

    def _count_grid_intersections(self, intersections: np.ndarray) -> Tuple[int, int]:
        if intersections is None or intersections.size == 0:
            return 0, 0
        mask_bin = (intersections > 0).astype(np.uint8)
        if mask_bin.sum() == 0:
            return 0, 0
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        return max(0, num_labels - 1), int(mask_bin.sum())

    def _choose_cuts_from_bands(self, bands, max_cuts, min_part_px, length, edge_margin_px):
        if not bands or max_cuts <= 0:
            return []
        bands = sorted(
            bands,
            key=lambda b: (-(b[1] - b[0]), abs((b[0] + b[1]) / 2 - length / 2))
        )
        cuts = []
        for b0, b1 in bands:
            c = int(round((b0 + b1) / 2))
            if c <= edge_margin_px or c >= length - edge_margin_px:
                continue
            cuts.append(c)
            cuts = sorted(set(cuts))
            pts = [0] + cuts + [length]
            ok = all((pts[i + 1] - pts[i]) >= min_part_px for i in range(len(pts) - 1))
            if not ok:
                cuts.remove(c)
            if len(cuts) >= max_cuts:
                break
        return cuts

    def _merge_uniform_cuts(self, cuts, target_cuts, length):
        if target_cuts <= 0:
            return []
        uniform = [int(round(length * i / (target_cuts + 1))) for i in range(1, target_cuts + 1)]
        merged = sorted(set(cuts + uniform))
        if len(merged) < target_cuts:
            return uniform[:target_cuts]
        if len(merged) <= target_cuts:
            return merged
        # 选择最接近均匀位置的切分点
        picked = []
        for u in uniform:
            nearest = min(merged, key=lambda c: abs(c - u))
            picked.append(nearest)
        picked = sorted(set(picked))
        if len(picked) < target_cuts:
            return uniform[:target_cuts]
        return picked[:target_cuts]

    def _find_line_bands(self, proj: np.ndarray, q_line: float, min_band_px: int):
        thr = np.quantile(proj, q_line)
        is_line = proj >= thr
        bands = []
        i = 0
        n = len(is_line)
        while i < n:
            if not is_line[i]:
                i += 1
                continue
            j = i
            while j < n and is_line[j]:
                j += 1
            if (j - i) >= min_band_px:
                bands.append((i, j))
            i = j
        return bands

    def _find_strong_line_bands(self, proj: np.ndarray, min_cover_px: int, min_band_px: int):
        is_line = proj >= min_cover_px
        bands = []
        i = 0
        n = len(is_line)
        while i < n:
            if not is_line[i]:
                i += 1
                continue
            j = i
            while j < n and is_line[j]:
                j += 1
            if (j - i) >= min_band_px:
                bands.append((i, j))
            i = j
        return bands

    def _snap_cuts_to_bands(self, cuts, bands, max_dist_px):
        if not cuts or not bands:
            return cuts
        centers = [int(round((b0 + b1) / 2)) for b0, b1 in bands]
        snapped = []
        for cut in cuts:
            nearest = min(centers, key=lambda c: abs(c - cut))
            if abs(nearest - cut) <= max_dist_px:
                snapped.append(nearest)
            else:
                snapped.append(cut)
        return sorted(set(snapped))

    def _infer_mask_cuts(self, mask_path: str, target_cols: int, target_rows: int, layout_type: str = "STRUCTURED"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask01 = (mask > 0).astype(np.uint8) * 255
        h, w = mask01.shape

        # 策略 A: TEXT_ONLY -> 找空白缝隙
        # 策略 B: STRUCTURED -> 找长直线
        if layout_type == "STRUCTURED":
            # 提取长直线（水平/垂直分离），使用更强的核抑制文字
            h_lines, v_lines = self._extract_long_lines(
                mask01,
                kx_ratio=MASK_LINE_KERNEL_X_RATIO_STRUCT,
                ky_ratio=MASK_LINE_KERNEL_Y_RATIO_STRUCT
            )
            find_func = self._find_line_bands
            q_x = MASK_PROJ_Q_LINE_X
            q_y = MASK_PROJ_Q_LINE_Y
        else:
            # 纯文字，使用原 mask
            h_lines, v_lines = mask01, mask01
            find_func = self._find_blank_bands
            q_x = MASK_PROJ_Q_BLANK_X
            q_y = MASK_PROJ_Q_BLANK_Y

        v_strength = (v_lines > 0).sum(axis=0)
        proj_x = v_strength
        min_band_x = max(3, int(round(w * MASK_PROJ_MIN_BAND_RATIO)))
        min_band_y = max(3, int(round(h * MASK_PROJ_MIN_BAND_RATIO)))
        min_part_x = max(50, int(round(w * MASK_PROJ_MIN_PART_RATIO)))
        min_part_y = max(50, int(round(h * MASK_PROJ_MIN_PART_RATIO)))
        edge_margin_x = int(round(w * MASK_PROJ_EDGE_MARGIN_RATIO))
        edge_margin_y = int(round(h * MASK_PROJ_EDGE_MARGIN_RATIO))

        if layout_type == "STRUCTURED":
            min_v_cover = int(round(h * STRUCT_VLINE_COVER_RATIO))
            bands_x = self._find_strong_line_bands(v_strength, min_v_cover, min_band_x)
            if not bands_x:
                bands_x = self._find_line_bands(v_strength, q_x, min_band_x)
        else:
            bands_x = find_func(proj_x, q_x, min_band_x)

        cuts_x = self._choose_cuts_from_bands(
            bands_x, max(0, target_cols - 1), min_part_x, w, edge_margin_x
        )
        if layout_type == "STRUCTURED" and bands_x:
            snap_dist = int(round(w * STRUCT_SNAP_MAX_RATIO))
            cuts_x = self._snap_cuts_to_bands(cuts_x, bands_x, snap_dist)
        if layout_type == "STRUCTURED" and not bands_x:
            # 结构图没有识别到明确竖线时，不强行补刀，避免切断文字
            cuts_x = []
        if target_cols > 1 and len(cuts_x) < target_cols - 1 and not (layout_type == "STRUCTURED" and not bands_x):
            cuts_x = self._merge_uniform_cuts(cuts_x, target_cols - 1, w)

        x_pts = [0] + sorted(cuts_x) + [w]
        cuts_y_per_col = []
        for i in range(len(x_pts) - 1):
            x0, x1 = x_pts[i], x_pts[i + 1]
            if x1 <= x0:
                cuts_y_per_col.append([])
                continue
            h_strength = (h_lines[:, x0:x1] > 0).sum(axis=1)
            proj_y = h_strength
            if layout_type == "STRUCTURED":
                min_h_cover = int(round((x1 - x0) * STRUCT_HLINE_COVER_RATIO))
                bands_y = self._find_strong_line_bands(h_strength, min_h_cover, min_band_y)
                if not bands_y:
                    bands_y = self._find_line_bands(h_strength, q_y, min_band_y)
            else:
                bands_y = find_func(proj_y, q_y, min_band_y)
            cuts_y = self._choose_cuts_from_bands(
                bands_y, max(0, target_rows - 1), min_part_y, h, edge_margin_y
            )
            if layout_type == "STRUCTURED" and bands_y:
                snap_dist_y = int(round(h * STRUCT_SNAP_MAX_RATIO))
                cuts_y = self._snap_cuts_to_bands(cuts_y, bands_y, snap_dist_y)
            # 智能补刀：如果是结构图，且完全没找到横线，可能不需要切（如附栏）
            if layout_type == "STRUCTURED":
                if not bands_y:
                    cuts_y = []
                # 如果没找到任何线，且本来就只要切少数几刀，就不补刀了
                if not cuts_y and target_rows <= 2:
                    pass
                elif len(cuts_y) < target_rows - 1:
                    cuts_y = self._merge_uniform_cuts(cuts_y, target_rows - 1, h)
            else:
                # 纯文字模式，如果没找到空白带，还是要补刀（防止图太大）
                if len(cuts_y) < target_rows - 1:
                    cuts_y = self._merge_uniform_cuts(cuts_y, target_rows - 1, h)
            
            cuts_y_per_col.append(cuts_y)

        return cuts_x, cuts_y_per_col, w, h

    def _build_boundaries_from_mask(self, mask_path: str, frame_x_min, frame_x_max,
                                    frame_y_min, frame_y_max, scale,
                                    target_cols: int, target_rows: int, layout_type: str = "STRUCTURED"):
        inferred = self._infer_mask_cuts(mask_path, target_cols, target_rows, layout_type)
        if not inferred:
            return None
        cuts_x_px, cuts_y_px_per_col, w, h = inferred

        def px_to_dxf_x(px):
            return frame_x_min + px / scale

        def px_to_dxf_y(px):
            return frame_y_max - px / scale

        col_boundaries = [frame_x_min]
        for cut_px in sorted(cuts_x_px):
            col_boundaries.append(px_to_dxf_x(cut_px))
        col_boundaries.append(frame_x_max)

        col_boundaries = sorted(set(col_boundaries))
        row_boundaries_per_col = []
        for cuts_y_px in cuts_y_px_per_col:
            row_boundaries = [frame_y_max]
            for cut_px in sorted(cuts_y_px):
                row_boundaries.append(px_to_dxf_y(cut_px))
            row_boundaries.append(frame_y_min)
            row_boundaries = sorted(set(row_boundaries), reverse=True)
            row_boundaries_per_col.append(row_boundaries)

        return col_boundaries, row_boundaries_per_col, cuts_x_px, cuts_y_px_per_col
    
    def _get_font(self, size, is_chinese):
        scale = CHINESE_FONT_SCALE if is_chinese else WESTERN_FONT_SCALE
        size = max(1, int(size * scale))
        key = (size, is_chinese)
        if key not in self._font_cache:
            path = self.chinese_font if is_chinese else self.western_font
            try:
                self._font_cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            except:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]
    
    def _replace_cad_symbols(self, text: str) -> str:
        result = text
        for code in sorted(CAD_SYMBOL_TABLE.keys(), key=len, reverse=True):
            result = result.replace(code, CAD_SYMBOL_TABLE[code])
        return result
    
    def _get_entity_color(self, entity, parent_layer: str = None) -> Tuple[int, int, int]:
        try:
            color_index = entity.dxf.color
            if color_index == 256:
                layer_name = getattr(entity.dxf, 'layer', parent_layer) or parent_layer or '0'
                return self.layer_colors.get(layer_name, (255, 255, 255))
            if color_index == 0:
                return (255, 255, 255)
            return self._aci_to_rgb(color_index)
        except:
            return (255, 255, 255)
    
    def find_frames(self) -> List[FrameInfo]:
        """查找DXF中的所有图框
        
        支持两种方式：
        1. 标准图框块引用（如民用院-A1H）
        2. 组合块（整个图纸在一个块内）
        """
        print("\n查找图框...")
        frames = []
        
        # 方法1: 尝试标准图框块
        block = self.doc.blocks.get(self.frame_block_name)
        if block:
            block_outer = None
            for entity in block:
                if entity.dxftype() == 'LWPOLYLINE' and entity.closed:
                    points = list(entity.get_points('xy'))
                    if len(points) == 4:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        w = max(xs) - min(xs)
                        if w > 50000:
                            block_outer = (min(xs), min(ys), max(xs), max(ys))
                            break
            
            if block_outer:
                print(f"  找到标准图框块: {self.frame_block_name}")
                print(f"  块内图框: {block_outer[2]-block_outer[0]:.0f} x {block_outer[3]-block_outer[1]:.0f}")
                
                for insert in self.msp.query('INSERT'):
                    if insert.dxf.name == self.frame_block_name:
                        ix, iy = insert.dxf.insert.x, insert.dxf.insert.y
                        sx = getattr(insert.dxf, 'xscale', 1.0)
                        sy = getattr(insert.dxf, 'yscale', 1.0)
                        frames.append(FrameInfo(
                            index=len(frames) + 1,
                            x_min=ix + block_outer[0] * sx,
                            y_min=iy + block_outer[1] * sy,
                            x_max=ix + block_outer[2] * sx,
                            y_max=iy + block_outer[3] * sy,
                            insert_x=ix, insert_y=iy,
                            scale_x=sx, scale_y=sy
                        ))
        
        # 方法2: 如果没找到标准图框，检查组合块
        if not frames:
            print(f"  未找到标准图框块'{self.frame_block_name}'，尝试识别组合块...")
            
            for entity in self.msp:
                if entity.dxftype() == 'INSERT':
                    block_name = entity.dxf.name
                    blk = self.doc.blocks.get(block_name)
                    if not blk:
                        continue
                    
                    ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
                    sx = getattr(entity.dxf, 'xscale', 1.0)
                    sy = getattr(entity.dxf, 'yscale', 1.0)
                    
                    # 递归计算块内所有实体的边界（包括嵌套块）
                    all_x = []
                    all_y = []
                    self._collect_block_bounds_recursive(blk, ix, iy, sx, sy, all_x, all_y, 0)
                    
                    if all_x and all_y:
                        # 过滤异常值：使用文字边界作为主要参考
                        text_x = []
                        text_y = []
                        self._collect_texts_for_filter(blk, ix, iy, sx, sy, text_x, text_y, 0)
                        
                        # 如果有文字，用文字边界来过滤异常坐标
                        if text_x and text_y:
                            tx_min, tx_max = min(text_x), max(text_x)
                            ty_min, ty_max = min(text_y), max(text_y)
                            margin = 5000  # 允许边界比文字范围大5000mm
                            
                            filtered_x = [x for x in all_x if tx_min - margin <= x <= tx_max + margin]
                            filtered_y = [y for y in all_y if ty_min - margin <= y <= ty_max + margin]
                            
                            if filtered_x and filtered_y:
                                all_x = filtered_x
                                all_y = filtered_y
                        
                        w = max(all_x) - min(all_x)
                        h = max(all_y) - min(all_y)
                        
                        # 如果尺寸足够大，认为是一个完整的图框
                        if w > 50000 and h > 30000:
                            # 更新图框块名
                            self.frame_block_name = block_name
                            
                            print(f"  找到组合块: {block_name}")
                            print(f"  世界坐标边界: X={min(all_x):.0f}~{max(all_x):.0f}, Y={min(all_y):.0f}~{max(all_y):.0f}")
                            print(f"  尺寸: {w:.0f} x {h:.0f}")
                            
                            frames.append(FrameInfo(
                                index=len(frames) + 1,
                                x_min=min(all_x),
                                y_min=min(all_y),
                                x_max=max(all_x),
                                y_max=max(all_y),
                                insert_x=ix, insert_y=iy,
                                scale_x=sx, scale_y=sy
                            ))
        
        # 按位置排序
        frames.sort(key=lambda f: (-f.y_min, f.x_min))
        for i, f in enumerate(frames, 1):
            f.index = i
            print(f"  图框 #{i}: ({f.x_min:.0f}, {f.y_min:.0f}), 尺寸: {f.x_max-f.x_min:.0f}x{f.y_max-f.y_min:.0f}")
        
        print(f"\n找到 {len(frames)} 个图框")
        return frames
    
    def _collect_block_bounds_recursive(self, block, offset_x, offset_y, scale_x, scale_y, 
                                         all_x, all_y, depth=0):
        """递归收集块内所有实体的边界（包括嵌套块）"""
        if depth > 10:
            return
        
        for bent in block:
            try:
                etype = bent.dxftype()
                if etype == 'TEXT':
                    x = bent.dxf.insert.x * scale_x + offset_x
                    y = bent.dxf.insert.y * scale_y + offset_y
                    all_x.append(x)
                    all_y.append(y)
                elif etype == 'MTEXT':
                    x = bent.dxf.insert.x * scale_x + offset_x
                    y = bent.dxf.insert.y * scale_y + offset_y
                    all_x.append(x)
                    all_y.append(y)
                elif etype == 'LINE':
                    x1 = bent.dxf.start.x * scale_x + offset_x
                    y1 = bent.dxf.start.y * scale_y + offset_y
                    x2 = bent.dxf.end.x * scale_x + offset_x
                    y2 = bent.dxf.end.y * scale_y + offset_y
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])
                elif etype == 'LWPOLYLINE':
                    points = list(bent.get_points('xy'))
                    for p in points:
                        x = p[0] * scale_x + offset_x
                        y = p[1] * scale_y + offset_y
                        all_x.append(x)
                        all_y.append(y)
                elif etype == 'INSERT':
                    nested_name = bent.dxf.name
                    nested_block = self.doc.blocks.get(nested_name)
                    if nested_block:
                        nix = bent.dxf.insert.x * scale_x + offset_x
                        niy = bent.dxf.insert.y * scale_y + offset_y
                        nsx = getattr(bent.dxf, 'xscale', 1.0) * scale_x
                        nsy = getattr(bent.dxf, 'yscale', 1.0) * scale_y
                        self._collect_block_bounds_recursive(nested_block, nix, niy, nsx, nsy,
                                                             all_x, all_y, depth + 1)
            except:
                pass
    
    def _collect_lines_recursive(self, block, offset_x, offset_y, scale_x, scale_y,
                                  h_segments, v_segments, depth=0):
        """递归收集块内所有线段（用于表格检测和切分）"""
        if depth > 10:
            return
        
        for bent in block:
            try:
                etype = bent.dxftype()
                if etype == 'LINE':
                    x1 = bent.dxf.start.x * scale_x + offset_x
                    y1 = bent.dxf.start.y * scale_y + offset_y
                    x2 = bent.dxf.end.x * scale_x + offset_x
                    y2 = bent.dxf.end.y * scale_y + offset_y
                    
                    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 1000:  # 忽略太短的线
                        if abs(y1 - y2) < 10:  # 水平线
                            h_segments.append({
                                'y': (y1+y2)/2, 'x_min': min(x1,x2), 
                                'x_max': max(x1,x2), 'length': abs(x2-x1)
                            })
                        elif abs(x1 - x2) < 10:  # 垂直线
                            v_segments.append({
                                'x': (x1+x2)/2, 'y_min': min(y1,y2), 
                                'y_max': max(y1,y2), 'length': abs(y2-y1)
                            })
                
                elif etype == 'LWPOLYLINE':
                    points = list(bent.get_points('xy'))
                    if bent.is_closed and len(points) >= 2:
                        points.append(points[0])
                    
                    for i in range(len(points) - 1):
                        x1 = points[i][0] * scale_x + offset_x
                        y1 = points[i][1] * scale_y + offset_y
                        x2 = points[i+1][0] * scale_x + offset_x
                        y2 = points[i+1][1] * scale_y + offset_y
                        
                        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if length > 1000:
                            if abs(y1 - y2) < 10:
                                h_segments.append({
                                    'y': (y1+y2)/2, 'x_min': min(x1,x2),
                                    'x_max': max(x1,x2), 'length': abs(x2-x1)
                                })
                            elif abs(x1 - x2) < 10:
                                v_segments.append({
                                    'x': (x1+x2)/2, 'y_min': min(y1,y2),
                                    'y_max': max(y1,y2), 'length': abs(y2-y1)
                                })
                
                elif etype == 'INSERT':
                    nested_name = bent.dxf.name
                    nested_block = self.doc.blocks.get(nested_name)
                    if nested_block:
                        nix = bent.dxf.insert.x * scale_x + offset_x
                        niy = bent.dxf.insert.y * scale_y + offset_y
                        nsx = getattr(bent.dxf, 'xscale', 1.0) * scale_x
                        nsy = getattr(bent.dxf, 'yscale', 1.0) * scale_y
                        self._collect_lines_recursive(nested_block, nix, niy, nsx, nsy,
                                                      h_segments, v_segments, depth + 1)
            except:
                pass
    
    def _detect_large_table(self, frame_x_min, frame_x_max, frame_y_min, frame_y_max):
        """检测是否存在大块表格
        
        大块表格定义：由线段组成的封闭矩形，宽或高达到图框的TABLE_SIZE_THRESHOLD以上
        从最外层开始检测，避免误判表格内的单元格
        
        返回：(是否存在大表格, 表格边界列表)
        """
        frame_width = frame_x_max - frame_x_min
        frame_height = frame_y_max - frame_y_min
        
        # 收集所有线段
        h_segments = []
        v_segments = []
        
        block = self.doc.blocks.get(self.frame_block_name)
        if block:
            for entity in self.msp:
                if entity.dxftype() == 'INSERT' and entity.dxf.name == self.frame_block_name:
                    ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
                    sx = getattr(entity.dxf, 'xscale', 1.0)
                    sy = getattr(entity.dxf, 'yscale', 1.0)
                    self._collect_lines_recursive(block, ix, iy, sx, sy, h_segments, v_segments, 0)
                    break
        
        # 按X坐标聚合垂直线
        x_groups = defaultdict(list)
        for seg in v_segments:
            if frame_x_min < seg['x'] < frame_x_max:
                x_key = round(seg['x'] / 100) * 100
                x_groups[x_key].append(seg)
        
        # 找长垂直线（高度达到阈值）
        long_verticals = []
        for x_key in sorted(x_groups.keys()):
            segments = x_groups[x_key]
            # 计算该X位置的总覆盖范围
            y_span = max(s['y_max'] for s in segments) - min(s['y_min'] for s in segments)
            
            if y_span > frame_height * TABLE_SIZE_THRESHOLD:
                long_verticals.append({
                    'x': x_key, 
                    'y_span': y_span,
                    'y_min': min(s['y_min'] for s in segments),
                    'y_max': max(s['y_max'] for s in segments)
                })
        
        # 按Y坐标聚合水平线
        y_groups = defaultdict(list)
        for seg in h_segments:
            if frame_y_min < seg['y'] < frame_y_max:
                y_key = round(seg['y'] / 100) * 100
                y_groups[y_key].append(seg)
        
        # 找长水平线（宽度达到阈值）
        long_horizontals = []
        for y_key in sorted(y_groups.keys()):
            segments = y_groups[y_key]
            x_span = max(s['x_max'] for s in segments) - min(s['x_min'] for s in segments)
            
            if x_span > frame_width * TABLE_SIZE_THRESHOLD:
                long_horizontals.append({
                    'y': y_key,
                    'x_span': x_span,
                    'x_min': min(s['x_min'] for s in segments),
                    'x_max': max(s['x_max'] for s in segments)
                })
        
        # 检测封闭矩形表格
        tables = []
        
        # 如果有足够的长垂直线和长水平线，可能存在大表格
        if len(long_verticals) >= 2 and len(long_horizontals) >= 2:
            # 找最外层的边界（从外向内检测）
            v_sorted = sorted(long_verticals, key=lambda v: v['x'])
            h_sorted = sorted(long_horizontals, key=lambda h: h['y'])
            
            # 尝试找最大的封闭矩形
            for i, v_left in enumerate(v_sorted[:-1]):
                for v_right in reversed(v_sorted[i+1:]):
                    for j, h_bottom in enumerate(h_sorted[:-1]):
                        for h_top in reversed(h_sorted[j+1:]):
                            # 检查是否形成封闭矩形
                            rect_width = v_right['x'] - v_left['x']
                            rect_height = h_top['y'] - h_bottom['y']
                            
                            # 检查尺寸是否达到阈值
                            if (rect_width > frame_width * TABLE_SIZE_THRESHOLD or 
                                rect_height > frame_height * TABLE_SIZE_THRESHOLD):
                                
                                # 检查四条边是否都存在（允许一定容差）
                                left_ok = abs(v_left['y_min'] - h_bottom['y']) < 500 or v_left['y_min'] < h_bottom['y']
                                right_ok = abs(v_right['y_min'] - h_bottom['y']) < 500 or v_right['y_min'] < h_bottom['y']
                                bottom_ok = abs(h_bottom['x_min'] - v_left['x']) < 500 or h_bottom['x_min'] < v_left['x']
                                top_ok = abs(h_top['x_min'] - v_left['x']) < 500 or h_top['x_min'] < v_left['x']
                                
                                if left_ok and right_ok and bottom_ok and top_ok:
                                    tables.append({
                                        'x_min': v_left['x'],
                                        'x_max': v_right['x'],
                                        'y_min': h_bottom['y'],
                                        'y_max': h_top['y'],
                                        'width': rect_width,
                                        'height': rect_height
                                    })
        
        # 去重并选择最大的表格
        if tables:
            # 按面积排序，选最大的
            tables.sort(key=lambda t: t['width'] * t['height'], reverse=True)
            # 去除被包含的小表格
            unique_tables = []
            for t in tables:
                is_contained = False
                for ut in unique_tables:
                    if (t['x_min'] >= ut['x_min'] - 100 and t['x_max'] <= ut['x_max'] + 100 and
                        t['y_min'] >= ut['y_min'] - 100 and t['y_max'] <= ut['y_max'] + 100):
                        is_contained = True
                        break
                if not is_contained:
                    unique_tables.append(t)
            
            if unique_tables:
                print(f"    检测到 {len(unique_tables)} 个大块表格:")
                for i, t in enumerate(unique_tables):
                    print(f"      表格{i+1}: X={t['x_min']:.0f}~{t['x_max']:.0f}, "
                          f"Y={t['y_min']:.0f}~{t['y_max']:.0f}, "
                          f"尺寸={t['width']:.0f}x{t['height']:.0f}")
                return True, unique_tables, v_segments, h_segments
        
        return False, [], v_segments, h_segments
    
    def _find_table_vertical_cut_lines(self, frame, v_segments, tables):
        """基于表格边界线查找纵向切分线（v12逻辑）"""
        frame_height = frame.y_max - frame.y_min
        frame_width = frame.x_max - frame.x_min
        
        # 按X聚合垂直线
        x_groups = defaultdict(list)
        for seg in v_segments:
            if frame.x_min < seg['x'] < frame.x_max:
                x_key = round(seg['x'] / 100) * 100
                x_groups[x_key].append(seg)
        
        # 找长垂直线
        long_verticals = []
        for x_key in sorted(x_groups.keys()):
            segments = x_groups[x_key]
            y_span = max(s['y_max'] for s in segments) - min(s['y_min'] for s in segments)
            
            if y_span > frame_height * 0.5:
                long_verticals.append({'x': x_key, 'y_span': y_span})
        
        if len(long_verticals) < 3:
            return None
        
        print(f"    找到 {len(long_verticals)} 条长垂直线")
        
        # 收集文字位置（用于判断左侧空白）
        texts = []
        block = self.doc.blocks.get(self.frame_block_name)
        if block:
            for entity in self.msp:
                if entity.dxftype() == 'INSERT' and entity.dxf.name == self.frame_block_name:
                    ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
                    sx = getattr(entity.dxf, 'xscale', 1.0)
                    sy = getattr(entity.dxf, 'yscale', 1.0)
                    self._collect_texts_recursive(block, ix, iy, sx, sy, texts, 0)
                    break
        
        title_y = frame.y_min + frame_height * 0.9
        
        # 找成对边界（间距小且之间无文字）
        paired = []
        for i in range(len(long_verticals) - 1):
            v1, v2 = long_verticals[i], long_verticals[i + 1]
            gap = v2['x'] - v1['x']
            
            if 100 < gap < 1500:
                between = [t for t in texts if v1['x'] < t['x'] < v2['x'] and t['y'] < title_y]
                if len(between) == 0:
                    paired.append((v1['x'], v2['x']))
        
        print(f"    找到 {len(paired)} 对成对边界")
        
        # 确定切分候选线
        cut_candidates = []
        
        # 从成对边界取右边的
        for p in paired:
            cut_candidates.append(p[1])
        
        # 检查左侧空白的独立长线
        for v in long_verticals:
            left_texts = [t for t in texts if v['x'] - 500 <= t['x'] < v['x'] and t['y'] < title_y]
            right_texts = [t for t in texts if v['x'] <= t['x'] < v['x'] + 3000 and t['y'] < title_y]
            
            if len(left_texts) == 0 and len(right_texts) > 0:
                in_pair = any(abs(v['x'] - p[0]) < 200 or abs(v['x'] - p[1]) < 200 for p in paired)
                if not in_pair:
                    cut_candidates.append(v['x'])
        
        cut_candidates = sorted(set(cut_candidates))
        print(f"    候选切分线: {[f'{x:.0f}' for x in cut_candidates]}")
        
        # 过滤太靠近边界的
        filtered = [x for x in cut_candidates 
                    if x - frame.x_min > frame_width * 0.1 and frame.x_max - x > frame_width * 0.05]
        
        if not filtered:
            return None
        
        print(f"    过滤后切分线: {[f'{x:.0f}' for x in filtered]}")
        
        # 选择能产生较均匀列宽的切分线组合
        if len(filtered) <= TARGET_COLS - 1:
            final_cuts = filtered
        else:
            from itertools import combinations
            
            best_cuts = None
            best_variance = float('inf')
            
            for combo in combinations(filtered, TARGET_COLS - 1):
                cuts = sorted(combo)
                boundaries = [frame.x_min] + list(cuts) + [frame.x_max]
                widths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
                
                # 计算宽度方差
                avg_width = sum(widths) / len(widths)
                variance = sum((w - avg_width) ** 2 for w in widths) / len(widths)
                
                # 惩罚太窄的列（宽度<15%）
                min_width = min(widths)
                if min_width < frame_width * 0.15:
                    variance += (frame_width * 0.15 - min_width) ** 2 * 10
                
                if variance < best_variance:
                    best_variance = variance
                    best_cuts = cuts
            
            final_cuts = best_cuts if best_cuts else filtered[:TARGET_COLS - 1]
        
        return final_cuts
    
    def _collect_texts_recursive(self, block, offset_x, offset_y, scale_x, scale_y, texts, depth=0):
        """递归收集块内所有文字"""
        if depth > 10:
            return
        
        for bent in block:
            try:
                etype = bent.dxftype()
                if etype in ('TEXT', 'MTEXT'):
                    x = bent.dxf.insert.x * scale_x + offset_x
                    y = bent.dxf.insert.y * scale_y + offset_y
                    texts.append({'x': x, 'y': y})
                elif etype == 'INSERT':
                    nested_name = bent.dxf.name
                    nested_block = self.doc.blocks.get(nested_name)
                    if nested_block:
                        nix = bent.dxf.insert.x * scale_x + offset_x
                        niy = bent.dxf.insert.y * scale_y + offset_y
                        nsx = getattr(bent.dxf, 'xscale', 1.0) * scale_x
                        nsy = getattr(bent.dxf, 'yscale', 1.0) * scale_y
                        self._collect_texts_recursive(nested_block, nix, niy, nsx, nsy, texts, depth + 1)
            except:
                pass

    def _collect_texts_for_filter(self, block, offset_x, offset_y, scale_x, scale_y, text_x, text_y, depth=0):
        """递归收集块内文字坐标（用于过滤异常坐标）"""
        if depth > 10:
            return
        
        for bent in block:
            try:
                etype = bent.dxftype()
                if etype in ('TEXT', 'MTEXT'):
                    x = bent.dxf.insert.x * scale_x + offset_x
                    y = bent.dxf.insert.y * scale_y + offset_y
                    text_x.append(x)
                    text_y.append(y)
                elif etype == 'INSERT':
                    nested_name = bent.dxf.name
                    nested_block = self.doc.blocks.get(nested_name)
                    if nested_block:
                        nix = bent.dxf.insert.x * scale_x + offset_x
                        niy = bent.dxf.insert.y * scale_y + offset_y
                        nsx = getattr(bent.dxf, 'xscale', 1.0) * scale_x
                        nsy = getattr(bent.dxf, 'yscale', 1.0) * scale_y
                        self._collect_texts_for_filter(nested_block, nix, niy, nsx, nsy, text_x, text_y, depth + 1)
            except:
                pass
    
    def _find_table_horizontal_cut_lines(self, col_left, col_right, col_y_min, col_y_max, h_segments):
        """查找横向切分线（v12逻辑：优先长度，保证间距）"""
        col_width = col_right - col_left
        col_height = col_y_max - col_y_min
        
        if col_width < 3000 or col_height < 5000:
            return []
        
        # 聚合水平线
        y_groups = defaultdict(list)
        for seg in h_segments:
            y_key = round(seg['y'] / 50) * 50
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(seg)
        
        # 收集候选线
        candidates = []
        for y_key, segments in y_groups.items():
            for seg in segments:
                overlap = min(seg['x_max'], col_right) - max(seg['x_min'], col_left)
                coverage = overlap / col_width if col_width > 0 else 0
                
                if coverage > 0.3:
                    # 排除边框
                    if (seg['y'] - col_y_min) > col_height * 0.05 and (col_y_max - seg['y']) > col_height * 0.05:
                        candidates.append({
                            'y': seg['y'], 'overlap': overlap, 'coverage': coverage,
                            'x_min': seg['x_min'], 'x_max': seg['x_max']
                        })
        
        if not candidates:
            return []
        
        # 去重
        unique = {}
        for h in candidates:
            y_key = round(h['y'] / 300) * 300
            if y_key not in unique or h['overlap'] > unique[y_key]['overlap']:
                unique[y_key] = h
        candidates = sorted(unique.values(), key=lambda c: -c['coverage'])
        
        # 选择：按覆盖率从高到低，保证最小间距
        min_gap = col_height * 0.2
        selected = []
        
        for c in candidates:
            too_close = False
            for s in selected:
                if abs(c['y'] - s['y']) < min_gap:
                    too_close = True
                    break
            if (c['y'] - col_y_min) < min_gap or (col_y_max - c['y']) < min_gap:
                too_close = True
            
            if not too_close:
                selected.append(c)
                if len(selected) >= TARGET_ROWS - 1:
                    break
        
        # 如果选不够，放宽间距
        if len(selected) < TARGET_ROWS - 1:
            min_gap = col_height * 0.15
            for c in candidates:
                if c in selected:
                    continue
                too_close = any(abs(c['y'] - s['y']) < min_gap for s in selected)
                if not too_close:
                    selected.append(c)
                    if len(selected) >= TARGET_ROWS - 1:
                        break
        
        return selected
    
    def _is_chinese(self, char):
        return '\u4e00' <= char <= '\u9fff'
    
    def _transform_point(self, x, y, insert_x, insert_y, scale_x, scale_y, rotation=0):
        x = x * scale_x
        y = y * scale_y
        if rotation != 0:
            rad = math.radians(rotation)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r
        return x + insert_x, y + insert_y
    
    def _collect_entity_positions(self, entities, insert_x, insert_y, scale_x, scale_y, 
                                   rotation, bounds, positions, depth=0):
        """收集实体的起始位置（用于确定切分点）"""
        if depth > 15:
            return
        
        x_min, y_min, x_max, y_max = bounds
        
        for entity in entities:
            try:
                etype = entity.dxftype()
                
                if etype == 'TEXT':
                    tx, ty = self._transform_point(
                        entity.dxf.insert.x, entity.dxf.insert.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    
                    if x_min <= tx <= x_max and y_min <= ty <= y_max:
                        positions.append({'x': tx, 'y': ty, 'type': 'text'})
                
                elif etype == 'MTEXT':
                    tx, ty = self._transform_point(
                        entity.dxf.insert.x, entity.dxf.insert.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    
                    if x_min <= tx <= x_max and y_min <= ty <= y_max:
                        positions.append({'x': tx, 'y': ty, 'type': 'text'})
                
                elif etype == 'INSERT':
                    block_name = entity.dxf.name
                    block = self.doc.blocks.get(block_name)
                    if block:
                        nix, niy = self._transform_point(
                            entity.dxf.insert.x, entity.dxf.insert.y,
                            insert_x, insert_y, scale_x, scale_y, rotation)
                        nsx = scale_x * getattr(entity.dxf, 'xscale', 1.0)
                        nsy = scale_y * getattr(entity.dxf, 'yscale', 1.0)
                        nrot = rotation + getattr(entity.dxf, 'rotation', 0)
                        
                        self._collect_entity_positions(block, nix, niy, nsx, nsy, nrot,
                                                       bounds, positions, depth + 1)
                
                elif etype in ('CIRCLE', 'ARC', 'ELLIPSE'):
                    cx, cy = self._transform_point(
                        entity.dxf.center.x, entity.dxf.center.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    r = entity.dxf.radius * abs(scale_x)
                    
                    # 记录圆/弧的最左点
                    left_x = cx - r
                    if x_min <= left_x <= x_max and y_min <= cy <= y_max:
                        positions.append({'x': left_x, 'y': cy, 'type': 'shape'})
                
            except:
                continue
    
    def _calculate_split_boundaries(self, positions, frame_x_min, frame_x_max,
                                     frame_y_min, frame_y_max, frame=None,
                                     cluster_only: bool = False):
        """基于文字起始位置密度计算切分边界
        
        逻辑顺序：
        1. 如果有手动切分点，直接使用
        2. 优先检测大块表格，如果存在则使用表格模式切分
        3. 否则使用文字密度聚类模式
        """
        if not cluster_only:
            # 1. 手动切分点
            if MANUAL_CUT_X:
                print(f"    使用手动切分点: {MANUAL_CUT_X}")
                col_boundaries = [frame_x_min] + list(MANUAL_CUT_X) + [frame_x_max]
                # 收集线段用于横向切分
                h_segments = []
                v_segments = []
                block = self.doc.blocks.get(self.frame_block_name)
                if block and frame:
                    for entity in self.msp:
                        if entity.dxftype() == 'INSERT' and entity.dxf.name == self.frame_block_name:
                            ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
                            sx = getattr(entity.dxf, 'xscale', 1.0)
                            sy = getattr(entity.dxf, 'yscale', 1.0)
                            self._collect_lines_recursive(block, ix, iy, sx, sy, h_segments, v_segments, 0)
                            break
                row_boundaries_per_col = self._calculate_table_row_boundaries(
                    col_boundaries, frame_y_min, frame_y_max, h_segments)
                return col_boundaries, row_boundaries_per_col

            # 2. 如果启用表格模式，优先检测大块表格
            if ENABLE_TABLE_MODE:
                has_table, tables, v_segments, h_segments = self._detect_large_table(
                    frame_x_min, frame_x_max, frame_y_min, frame_y_max)
                
                if has_table and frame:
                    print(f"    检测到大块表格，启用表格模式切分...")
                    # 使用表格边界线切分
                    table_cuts = self._find_table_vertical_cut_lines(frame, v_segments, tables)
                    if table_cuts:
                        print(f"    表格模式纵向切分线: {[f'{x:.0f}' for x in table_cuts]}")
                        col_boundaries = [frame_x_min] + table_cuts + [frame_x_max]
                        row_boundaries_per_col = self._calculate_table_row_boundaries(
                            col_boundaries, frame_y_min, frame_y_max, h_segments)
                        return col_boundaries, row_boundaries_per_col
                    else:
                        print(f"    表格模式未找到有效切分线，回退到聚类模式")
            else:
                print(f"    表格模式已禁用，使用文字聚类模式")
        else:
            print(f"    聚类模式(忽略表格/手动切分)")
        
        # 3. 文字密度聚类模式（原有逻辑）- 只关注文字，不受表格竖线影响
        # 收集所有文字的起始X位置，按100mm分桶统计
        x_counts = defaultdict(int)
        
        # 收集模型空间的文字
        for entity in self.msp:
            try:
                etype = entity.dxftype()
                if etype in ('TEXT', 'MTEXT'):
                    x = entity.dxf.insert.x
                    y = entity.dxf.insert.y
                    if frame_x_min <= x <= frame_x_max and frame_y_min <= y <= frame_y_max:
                        bucket = round(x / 100) * 100
                        x_counts[bucket] += 1
            except:
                continue
        
        # 收集块内的文字（递归处理嵌套块）
        block = self.doc.blocks.get(self.frame_block_name)
        if block:
            for entity in self.msp:
                if entity.dxftype() == 'INSERT' and entity.dxf.name == self.frame_block_name:
                    ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
                    sx = getattr(entity.dxf, 'xscale', 1.0)
                    sy = getattr(entity.dxf, 'yscale', 1.0)
                    
                    # 使用递归方法收集文字位置
                    texts = []
                    self._collect_texts_recursive(block, ix, iy, sx, sy, texts, 0)
                    for t in texts:
                        if frame_x_min <= t['x'] <= frame_x_max and frame_y_min <= t['y'] <= frame_y_max:
                            bucket = round(t['x'] / 100) * 100
                            x_counts[bucket] += 1
                    break
        
        if not x_counts:
            col_width = (frame_x_max - frame_x_min) / TARGET_COLS
            col_boundaries = [frame_x_min + i * col_width for i in range(TARGET_COLS + 1)]
            row_boundaries_per_col = [[frame_y_max - j * (frame_y_max - frame_y_min) / TARGET_ROWS 
                                       for j in range(TARGET_ROWS + 1)] for _ in range(TARGET_COLS)]
            return col_boundaries, row_boundaries_per_col
        
        # 找出密集聚类
        sorted_buckets = sorted(x_counts.keys())
        clusters = []
        current_cluster = {'start': sorted_buckets[0], 'end': sorted_buckets[0], 
                          'count': x_counts[sorted_buckets[0]], 'buckets': [sorted_buckets[0]]}
        
        for bucket in sorted_buckets[1:]:
            if bucket - current_cluster['end'] <= 2000:
                current_cluster['end'] = bucket
                current_cluster['count'] += x_counts[bucket]
                current_cluster['buckets'].append(bucket)
            else:
                clusters.append(current_cluster)
                current_cluster = {'start': bucket, 'end': bucket,
                                  'count': x_counts[bucket], 'buckets': [bucket]}
        clusters.append(current_cluster)
        
        # 按文字数量排序，取最大的TARGET_COLS个聚类
        clusters.sort(key=lambda c: -c['count'])
        top_clusters = clusters[:TARGET_COLS]
        
        # 按位置排序
        top_clusters.sort(key=lambda c: c['start'])
        
        print(f"    找到的主要聚类:")
        for i, c in enumerate(top_clusters):
            print(f"      列{i+1}: X={c['start']:.0f}~{c['end']:.0f}, 文字数={c['count']}")
        
        # 构建列边界
        col_boundaries = [frame_x_min]
        for i in range(1, len(top_clusters)):
            cut_x = top_clusters[i]['start'] - CUT_OFFSET_X
            col_boundaries.append(cut_x)
        col_boundaries.append(frame_x_max)
        
        # 计算行边界
        row_boundaries_per_col = self._calculate_row_boundaries(
            col_boundaries, frame_y_min, frame_y_max)
        
        return col_boundaries, row_boundaries_per_col
    
    def _draw_debug_lines(self, draw, image, frame_x_min, frame_x_max,
                          frame_y_min, frame_y_max, scale, col_boundaries,
                          x_starts_unique, mask_cuts_px=None):
        """绘制调试线条：文字起始位置和切分线"""
        
        img_height = image.height
        
        def dxf_x_to_img(dxf_x):
            return int((dxf_x - frame_x_min) * scale)
        
        def dxf_y_to_img(dxf_y):
            return int((frame_y_max - dxf_y) * scale)
        
        # 绘制切分线（红色粗线）并标注坐标
        font = self._get_font(24, False)
        for i, cut_x in enumerate(col_boundaries[1:-1], 1):  # 跳过首尾边界
            img_x = dxf_x_to_img(cut_x)
            if 0 <= img_x < image.width:
                # 画红色粗线
                draw.line([(img_x, 0), (img_x, img_height)], fill=(255, 0, 0), width=6)
                # 标注X坐标
                label = f"CUT{i}: X={cut_x:.0f}"
                draw.text((img_x + 5, 30 + i * 40), label, font=font, fill=(255, 0, 0))

        # 绘制 mask 投影切线（黄色）
        if mask_cuts_px:
            mask_cuts_x, mask_cuts_y_per_col = mask_cuts_px
            for i, cut_x in enumerate(sorted(mask_cuts_x), 1):
                if 0 <= cut_x < image.width:
                    draw.line([(cut_x, 0), (cut_x, img_height)], fill=(255, 200, 0), width=5)
                    label = f"MASK_X{i}"
                    draw.text((cut_x + 5, 10 + i * 30), label, font=font, fill=(255, 200, 0))
            x_pts = [0] + sorted(mask_cuts_x) + [image.width]
            for col_idx, cuts_y in enumerate(mask_cuts_y_per_col):
                if col_idx >= len(x_pts) - 1:
                    break
                x0, x1 = x_pts[col_idx], x_pts[col_idx + 1]
                for i, cut_y in enumerate(sorted(cuts_y), 1):
                    if 0 <= cut_y < image.height:
                        draw.line([(x0, cut_y), (x1, cut_y)], fill=(255, 200, 0), width=5)
                        label = f"MASK_Y{col_idx+1}-{i}"
                        draw.text((x0 + 5, cut_y + 5), label, font=font, fill=(255, 200, 0))
    
    def _collect_text_x_starts(self, frame_x_min, frame_x_max, frame_y_min, frame_y_max):
        """收集所有文字的起始X坐标（包括块内的文字）"""
        x_starts = []
        
        # 收集模型空间的文字
        for entity in self.msp:
            try:
                etype = entity.dxftype()
                if etype in ('TEXT', 'MTEXT'):
                    x = entity.dxf.insert.x
                    y = entity.dxf.insert.y
                    if frame_x_min <= x <= frame_x_max and frame_y_min <= y <= frame_y_max:
                        x_starts.append(x)
            except:
                continue
        
        # 收集块内的文字
        block = self.doc.blocks.get(self.frame_block_name)
        if block:
            # 找到对应的INSERT获取偏移
            for entity in self.msp:
                if entity.dxftype() == 'INSERT' and entity.dxf.name == self.frame_block_name:
                    ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
                    sx = getattr(entity.dxf, 'xscale', 1.0)
                    sy = getattr(entity.dxf, 'yscale', 1.0)
                    
                    for bent in block:
                        try:
                            if bent.dxftype() in ('TEXT', 'MTEXT'):
                                bx = bent.dxf.insert.x * sx + ix
                                by = bent.dxf.insert.y * sy + iy
                                if frame_x_min <= bx <= frame_x_max and frame_y_min <= by <= frame_y_max:
                                    x_starts.append(bx)
                        except:
                            continue
                    break
        
        return sorted(set(x_starts))

    def _calculate_row_boundaries(self, col_boundaries, frame_y_min, frame_y_max):
        """计算每列的行边界"""
        row_boundaries_per_col = []
        
        for col_idx in range(len(col_boundaries) - 1):
            col_x_min = col_boundaries[col_idx]
            col_x_max = col_boundaries[col_idx + 1]
            
            # 收集该列内文字的起始Y位置
            y_starts = []
            for entity in self.msp:
                try:
                    etype = entity.dxftype()
                    if etype == 'TEXT':
                        x = entity.dxf.insert.x
                        y = entity.dxf.insert.y
                        if col_x_min <= x < col_x_max and frame_y_min <= y <= frame_y_max:
                            y_starts.append(y)
                    elif etype == 'MTEXT':
                        x = entity.dxf.insert.x
                        y = entity.dxf.insert.y
                        if col_x_min <= x < col_x_max and frame_y_min <= y <= frame_y_max:
                            y_starts.append(y)
                except:
                    continue
            
            row_height = (frame_y_max - frame_y_min) / TARGET_ROWS
            row_boundaries = [frame_y_max]
            
            if y_starts:
                y_starts.sort(reverse=True)  # 从大到小（从上到下）
                
                for j in range(1, TARGET_ROWS):
                    # 下一行大约从这个Y位置开始
                    approx_row_start = frame_y_max - j * row_height
                    
                    search_min = approx_row_start - row_height * 0.3
                    search_max = approx_row_start + row_height * 0.3
                    
                    # 找这个范围内最上面的文字Y位置
                    nearby_ys = [y for y in y_starts if search_min <= y <= search_max]
                    
                    if nearby_ys:
                        next_row_top = max(nearby_ys)
                        cut_y = next_row_top + CUT_OFFSET_Y
                    else:
                        cut_y = approx_row_start
                    
                    row_boundaries.append(cut_y)
            else:
                for j in range(1, TARGET_ROWS):
                    row_boundaries.append(frame_y_max - j * row_height)
            
            row_boundaries.append(frame_y_min)
            row_boundaries_per_col.append(row_boundaries)
        
        return row_boundaries_per_col
    
    def _calculate_table_row_boundaries(self, col_boundaries, frame_y_min, frame_y_max, h_segments):
        """计算表格模式的行边界（使用水平线）"""
        row_boundaries_per_col = []
        
        for col_idx in range(len(col_boundaries) - 1):
            col_x_min = col_boundaries[col_idx]
            col_x_max = col_boundaries[col_idx + 1]
            
            # 使用水平线切分
            row_cuts = self._find_table_horizontal_cut_lines(
                col_x_min, col_x_max, frame_y_min, frame_y_max, h_segments)
            
            row_boundaries = [frame_y_max]
            for cut in sorted(row_cuts, key=lambda c: -c['y']):
                row_boundaries.append(cut['y'])
            row_boundaries.append(frame_y_min)
            
            # 如果没找到足够的切分线，用等分
            default_rows = TARGET_ROWS if TARGET_ROWS is not None else 3
            if len(row_boundaries) < default_rows + 1:
                row_height = (frame_y_max - frame_y_min) / default_rows
                row_boundaries = [frame_y_max]
                for j in range(1, default_rows):
                    row_boundaries.append(frame_y_max - j * row_height)
                row_boundaries.append(frame_y_min)
            
            row_boundaries_per_col.append(row_boundaries)
        
        return row_boundaries_per_col
    
    def _is_region_empty(self, image, x1, y1, x2, y2):
        """检查区域是否为空"""
        if x2 <= x1 or y2 <= y1:
            return True
        
        sample_step = max(1, min((x2 - x1) // 20, (y2 - y1) // 20))
        pixels = image.load()
        
        for x in range(x1, min(x2, image.width), sample_step):
            for y in range(y1, min(y2, image.height), sample_step):
                pixel = pixels[x, y]
                if pixel != BACKGROUND_COLOR:
                    return False
        
        return True
    
    def _split_and_save_image(self, image, frame_dir, frame_index, 
                               col_boundaries, row_boundaries_per_col, scale, bounds):
        """切分并保存图像"""
        x_min, y_min, x_max, y_max = bounds
        
        def dxf_to_img_x(dxf_x):
            return int((dxf_x - x_min) * scale)
        
        def dxf_to_img_y(dxf_y):
            return int((y_max - dxf_y) * scale)
        
        saved_count = 0
        split_index = 1
        
        cols = len(col_boundaries) - 1
        
        for col_idx in range(cols):
            col_left_dxf = col_boundaries[col_idx]
            col_right_dxf = col_boundaries[col_idx + 1]
            
            col_left_px = dxf_to_img_x(col_left_dxf)
            col_right_px = dxf_to_img_x(col_right_dxf)
            
            row_boundaries = row_boundaries_per_col[col_idx]
            rows = len(row_boundaries) - 1
            
            for row_idx in range(rows):
                row_top_dxf = row_boundaries[row_idx]
                row_bottom_dxf = row_boundaries[row_idx + 1]
                
                row_top_px = dxf_to_img_y(row_top_dxf)
                row_bottom_px = dxf_to_img_y(row_bottom_dxf)
                
                # 确保坐标有效
                col_left_px = max(0, col_left_px)
                col_right_px = min(image.width, col_right_px)
                row_top_px = max(0, row_top_px)
                row_bottom_px = min(image.height, row_bottom_px)
                
                if col_right_px <= col_left_px or row_bottom_px <= row_top_px:
                    continue
                
                # 检查区域是否为空
                if self._is_region_empty(image, col_left_px, row_top_px, 
                                         col_right_px, row_bottom_px):
                    continue
                
                # 裁剪
                crop_box = (col_left_px, row_top_px, col_right_px, row_bottom_px)
                cropped = image.crop(crop_box)
                
                # 保存
                split_path = os.path.join(frame_dir, f"frame_{frame_index}_{col_idx + 1}_{row_idx + 1}.png")
                cropped.save(split_path, 'PNG', dpi=(OUTPUT_DPI, OUTPUT_DPI))
                
                saved_count += 1
                split_index += 1
        
        return saved_count
    
    def _draw_block(self, draw, image, block_name, 
                    insert_x, insert_y, scale_x, scale_y, rotation,
                    to_img, line_width, bounds, scale,
                    text_data, drawn_boxes, parent_layer, depth):
        """递归绘制块内实体"""
        if depth > 15:
            return
        
        block = self.doc.blocks.get(block_name)
        if not block:
            return
        
        x_min, y_min, x_max, y_max = bounds
        
        for entity in block:
            try:
                etype = entity.dxftype()
                layer = getattr(entity.dxf, 'layer', parent_layer) or parent_layer
                color = self._get_entity_color(entity, layer)
                
                if etype == 'LINE':
                    sx, sy = self._transform_point(
                        entity.dxf.start.x, entity.dxf.start.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    ex, ey = self._transform_point(
                        entity.dxf.end.x, entity.dxf.end.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    
                    if max(sx, ex) < x_min or min(sx, ex) > x_max:
                        continue
                    if max(sy, ey) < y_min or min(sy, ey) > y_max:
                        continue
                    
                    draw.line([to_img(sx, sy), to_img(ex, ey)], fill=color, width=line_width)
                
                elif etype == 'LWPOLYLINE':
                    points = list(entity.get_points('xy'))
                    if not points:
                        continue
                    
                    world_pts = [self._transform_point(p[0], p[1], insert_x, insert_y, 
                                                       scale_x, scale_y, rotation) for p in points]
                    xs = [p[0] for p in world_pts]
                    ys = [p[1] for p in world_pts]
                    
                    if max(xs) < x_min or min(xs) > x_max:
                        continue
                    if max(ys) < y_min or min(ys) > y_max:
                        continue
                    
                    img_pts = [to_img(p[0], p[1]) for p in world_pts]
                    if len(img_pts) > 1:
                        draw.line(img_pts, fill=color, width=line_width)
                        if entity.closed and len(img_pts) > 2:
                            draw.line([img_pts[-1], img_pts[0]], fill=color, width=line_width)
                
                elif etype == 'POLYLINE':
                    try:
                        vertices = list(entity.vertices)
                        if not vertices:
                            continue
                        world_pts = [self._transform_point(
                            v.dxf.location.x, v.dxf.location.y,
                            insert_x, insert_y, scale_x, scale_y, rotation) for v in vertices]
                        
                        img_pts = [to_img(p[0], p[1]) for p in world_pts]
                        if len(img_pts) > 1:
                            draw.line(img_pts, fill=color, width=line_width)
                            if entity.is_closed and len(img_pts) > 2:
                                draw.line([img_pts[-1], img_pts[0]], fill=color, width=line_width)
                    except:
                        pass
                
                elif etype == 'CIRCLE':
                    cx, cy = self._transform_point(
                        entity.dxf.center.x, entity.dxf.center.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    r = entity.dxf.radius * abs(scale_x)
                    
                    if cx + r < x_min or cx - r > x_max:
                        continue
                    if cy + r < y_min or cy - r > y_max:
                        continue
                    
                    icx, icy = to_img(cx, cy)
                    ir = int(r * scale)
                    if ir > 0:
                        draw.ellipse([icx - ir, icy - ir, icx + ir, icy + ir],
                                    outline=color, width=line_width)
                
                elif etype == 'ARC':
                    cx, cy = self._transform_point(
                        entity.dxf.center.x, entity.dxf.center.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    r = entity.dxf.radius * abs(scale_x)
                    
                    if cx + r < x_min or cx - r > x_max:
                        continue
                    if cy + r < y_min or cy - r > y_max:
                        continue
                    
                    icx, icy = to_img(cx, cy)
                    ir = int(r * scale)
                    if ir > 0:
                        start = -entity.dxf.end_angle + rotation
                        end = -entity.dxf.start_angle + rotation
                        draw.arc([icx - ir, icy - ir, icx + ir, icy + ir],
                                start, end, fill=color, width=line_width)
                
                elif etype == 'TEXT':
                    tx, ty = self._transform_point(
                        entity.dxf.insert.x, entity.dxf.insert.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    
                    if tx < x_min - 1000 or tx > x_max + 1000:
                        continue
                    if ty < y_min - 1000 or ty > y_max + 1000:
                        continue
                    
                    text = self._replace_cad_symbols(entity.dxf.text)
                    height = entity.dxf.height * abs(scale_y)
                    text_rotation = getattr(entity.dxf, 'rotation', 0) + rotation
                    
                    if text.strip():
                        text_data.append({
                            'text': text.strip(), 'x': tx, 'y': ty,
                            'height': height, 'layer': layer
                        })
                        self._draw_text_compressed(draw, image, text, tx, ty, height,
                                                  text_rotation, color, bounds, scale, drawn_boxes)
                
                elif etype == 'MTEXT':
                    tx, ty = self._transform_point(
                        entity.dxf.insert.x, entity.dxf.insert.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    
                    if tx < x_min - 1000 or tx > x_max + 1000:
                        continue
                    if ty < y_min - 1000 or ty > y_max + 1000:
                        continue
                    
                    text = entity.text
                    text = re.sub(r'\\P', '\n', text)
                    text = re.sub(r'\\[A-Za-z][0-9.]*;?', '', text)
                    text = re.sub(r'[{}]', '', text)
                    text = self._replace_cad_symbols(text)
                    
                    height = getattr(entity.dxf, 'char_height', 2.5) * abs(scale_y)
                    text_rotation = getattr(entity.dxf, 'rotation', 0) + rotation
                    
                    if text.strip():
                        text_data.append({
                            'text': text.strip(), 'x': tx, 'y': ty,
                            'height': height, 'layer': layer
                        })
                        for i, line in enumerate(text.split('\n')):
                            if line.strip():
                                line_y = ty - i * height * 1.5
                                self._draw_text_compressed(draw, image, line, tx, line_y,
                                                          height, text_rotation, color, bounds,
                                                          scale, drawn_boxes)
                
                elif etype == 'INSERT':
                    nested_name = entity.dxf.name
                    nix, niy = self._transform_point(
                        entity.dxf.insert.x, entity.dxf.insert.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    nsx = scale_x * getattr(entity.dxf, 'xscale', 1.0)
                    nsy = scale_y * getattr(entity.dxf, 'yscale', 1.0)
                    nrot = rotation + getattr(entity.dxf, 'rotation', 0)
                    
                    self._draw_block(draw, image, nested_name,
                                    nix, niy, nsx, nsy, nrot,
                                    to_img, line_width, bounds, scale,
                                    text_data, drawn_boxes, layer, depth + 1)
                
                elif etype == 'SOLID':
                    try:
                        pts = [entity.dxf.vtx0, entity.dxf.vtx1, entity.dxf.vtx2]
                        if hasattr(entity.dxf, 'vtx3'):
                            pts.append(entity.dxf.vtx3)
                        
                        world_pts = [self._transform_point(p.x, p.y, insert_x, insert_y,
                                                           scale_x, scale_y, rotation) for p in pts]
                        img_pts = [to_img(p[0], p[1]) for p in world_pts]
                        
                        if len(img_pts) >= 3:
                            draw.polygon(img_pts, fill=color, outline=color)
                    except:
                        pass
                
                elif etype == 'POINT':
                    px, py = self._transform_point(
                        entity.dxf.location.x, entity.dxf.location.y,
                        insert_x, insert_y, scale_x, scale_y, rotation)
                    
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        ipx, ipy = to_img(px, py)
                        r = max(1, int(scale * 0.5))
                        draw.ellipse([ipx - r, ipy - r, ipx + r, ipy + r], fill=color)
                
                elif etype == 'ELLIPSE':
                    try:
                        cx, cy = self._transform_point(
                            entity.dxf.center.x, entity.dxf.center.y,
                            insert_x, insert_y, scale_x, scale_y, rotation)
                        
                        major = entity.dxf.major_axis
                        ratio = entity.dxf.ratio
                        a = math.sqrt(major.x**2 + major.y**2) * abs(scale_x)
                        b = a * ratio
                        
                        icx, icy = to_img(cx, cy)
                        ia, ib = int(a * scale), int(b * scale)
                        if ia > 0 and ib > 0:
                            draw.ellipse([icx - ia, icy - ib, icx + ia, icy + ib],
                                        outline=color, width=line_width)
                    except:
                        pass
                
                elif etype == 'SPLINE':
                    try:
                        ctrl_pts = list(entity.control_points)
                        if len(ctrl_pts) > 1:
                            world_pts = [self._transform_point(p[0], p[1], insert_x, insert_y,
                                                               scale_x, scale_y, rotation) 
                                        for p in ctrl_pts]
                            img_pts = [to_img(p[0], p[1]) for p in world_pts]
                            draw.line(img_pts, fill=color, width=line_width)
                    except:
                        pass
                
                elif etype == 'HATCH':
                    try:
                        for path in entity.paths:
                            if hasattr(path, 'vertices'):
                                pts = [(v[0], v[1]) for v in path.vertices]
                                if len(pts) > 1:
                                    world_pts = [self._transform_point(p[0], p[1], insert_x, insert_y,
                                                                       scale_x, scale_y, rotation) for p in pts]
                                    img_pts = [to_img(p[0], p[1]) for p in world_pts]
                                    draw.line(img_pts, fill=color, width=max(1, line_width // 2))
                                    if len(img_pts) > 2:
                                        draw.line([img_pts[-1], img_pts[0]], fill=color, width=max(1, line_width // 2))
                    except:
                        pass
                
            except:
                continue
    
    def _find_non_overlap_position(self, bbox, drawn_boxes, font_size, img_width):
        if not ENABLE_OVERLAP_PREVENTION:
            return bbox[0], bbox[1]
        return bbox[0], bbox[1]
    
    def _draw_text_compressed(self, draw, image, text, tx, ty, height, rotation,
                              color, bounds, scale, drawn_boxes):
        if not text.strip():
            return
        
        x_min, y_min, x_max, y_max = bounds
        font_size = int(height * scale)
        if font_size < 3:
            return
        
        base_x = (tx - x_min) * scale
        base_y = (y_max - ty) * scale
        
        segments = []
        curr_seg, curr_type = "", None
        for char in text:
            ctype = 'space' if char == ' ' else ('chinese' if self._is_chinese(char) else 'western')
            if ctype != curr_type and curr_seg:
                segments.append((curr_type, curr_seg))
                curr_seg = ""
            curr_seg += char
            curr_type = ctype
        if curr_seg:
            segments.append((curr_type, curr_seg))
        
        total_w = 0
        seg_info = []
        
        for stype, stext in segments:
            if stype == 'space':
                font = self._get_font(font_size, False)
                try:
                    w = font.getbbox(' ')[2] * len(stext)
                except:
                    w = font_size * 0.5 * len(stext)
                w *= TEXT_HORIZONTAL_SCALE
                seg_info.append({'type': stype, 'text': stext, 'width': w, 'font': font})
            
            elif stype == 'chinese':
                font = self._get_font(font_size, True)
                char_info = []
                seg_w = 0
                for c in stext:
                    try:
                        orig_w = font.getbbox(c)[2]
                    except:
                        orig_w = font_size
                    compressed_w = orig_w * CHINESE_COMPRESSION * TEXT_HORIZONTAL_SCALE
                    char_info.append({'char': c, 'orig_w': orig_w, 'compressed_w': compressed_w})
                    seg_w += compressed_w
                seg_info.append({'type': stype, 'text': stext, 'width': seg_w,
                                'font': font, 'chars': char_info})
            
            else:
                font = self._get_font(font_size, False)
                try:
                    w = font.getbbox(stext)[2]
                except:
                    w = font_size * len(stext) * 0.6
                w *= TEXT_HORIZONTAL_SCALE
                seg_info.append({'type': stype, 'text': stext, 'width': w, 'font': font,
                                'scale': TEXT_HORIZONTAL_SCALE})
            
            total_w += seg_info[-1]['width']
        
        bbox = [base_x, base_y - font_size, base_x + total_w, base_y + font_size * 0.3]
        new_x, new_y = self._find_non_overlap_position(bbox, drawn_boxes, font_size, image.width)
        
        offset_x = new_x - bbox[0]
        offset_y = new_y - bbox[1]
        base_x += offset_x
        base_y += offset_y
        
        curr_x = base_x
        draw_y = int(base_y - font_size * 0.8)
        
        for seg in seg_info:
            if seg['type'] == 'space':
                curr_x += seg['width']
            
            elif seg['type'] == 'chinese':
                for cinfo in seg['chars']:
                    char = cinfo['char']
                    orig_w = int(cinfo['orig_w'])
                    compressed_w = int(cinfo['compressed_w'])
                    
                    if orig_w <= 0 or compressed_w <= 0:
                        curr_x += max(1, compressed_w)
                        continue
                    
                    try:
                        char_h = int(font_size * 1.2)
                        tmp_img = Image.new('RGBA', (orig_w + 4, char_h + 4), (0, 0, 0, 0))
                        tmp_draw = ImageDraw.Draw(tmp_img)
                        tmp_draw.text((2, 2), char, font=seg['font'], fill=color + (255,))
                        
                        new_w = max(1, compressed_w)
                        compressed_img = tmp_img.resize((new_w, char_h + 4), Image.LANCZOS)
                        
                        paste_x = int(curr_x)
                        paste_y = int(draw_y - 2)
                        
                        if 0 <= paste_x < image.width and 0 <= paste_y < image.height:
                            image.paste(compressed_img, (paste_x, paste_y), compressed_img)
                    except:
                        try:
                            draw.text((int(curr_x), draw_y), char, font=seg['font'], fill=color)
                        except:
                            pass
                    
                    curr_x += compressed_w
            
            else:
                seg_scale = seg.get('scale', 1.0)
                if abs(seg_scale - 1.0) > 0.01:
                    try:
                        font = seg['font']
                        orig_w = int(seg['width'] / seg_scale)
                        char_h = int(font_size * 1.2)
                        
                        tmp_img = Image.new('RGBA', (orig_w + 4, char_h + 4), (0, 0, 0, 0))
                        tmp_draw = ImageDraw.Draw(tmp_img)
                        tmp_draw.text((2, 2), seg['text'], font=font, fill=color + (255,))
                        
                        new_w = max(1, int(seg['width']))
                        scaled_img = tmp_img.resize((new_w, char_h + 4), Image.LANCZOS)
                        
                        paste_x = int(curr_x)
                        paste_y = int(draw_y - 2)
                        
                        if 0 <= paste_x < image.width and 0 <= paste_y < image.height:
                            image.paste(scaled_img, (paste_x, paste_y), scaled_img)
                    except:
                        draw.text((int(curr_x), draw_y), seg['text'], font=seg['font'], fill=color)
                else:
                    try:
                        draw.text((int(curr_x), draw_y), seg['text'], font=seg['font'], fill=color)
                    except:
                        pass
                
                curr_x += seg['width']
    
    def render_frame(self, frame: FrameInfo):
        global TARGET_COLS, TARGET_ROWS
        print(f"\n渲染图框 #{frame.index}...")
        
        frame_dir = os.path.join(OUTPUT_DIR, f"frame_{frame.index}")
        os.makedirs(frame_dir, exist_ok=True)
        
        margin = 100
        x_min = frame.x_min - margin
        y_min = frame.y_min - margin
        x_max = frame.x_max + margin
        y_max = frame.y_max + margin
        bounds = (x_min, y_min, x_max, y_max)
        
        dxf_w, dxf_h = x_max - x_min, y_max - y_min
        
        base_scale = TARGET_DPI / 25.4
        img_w = int(dxf_w * base_scale)
        img_h = int(dxf_h * base_scale)
        
        if max(img_w, img_h) > MAX_IMAGE_DIMENSION:
            ratio = MAX_IMAGE_DIMENSION / max(img_w, img_h)
            img_w = int(img_w * ratio)
            img_h = int(img_h * ratio)
        
        scale = img_w / dxf_w
        print(f"  尺寸: {img_w}x{img_h}px, DPI: {OUTPUT_DPI}")
        
        def to_img(x, y):
            return int((x - x_min) * scale), int((y_max - y) * scale)
        
        image = Image.new('RGB', (img_w, img_h), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)
        text_data = []
        drawn_boxes = []
        line_width = max(1, int(scale * 0.18))
        
        # （切分边界计算移动到保存完整PNG之后进行）
        
        # 绘制图框块
        self._draw_block(draw, image, self.frame_block_name,
                        frame.insert_x, frame.insert_y,
                        frame.scale_x, frame.scale_y, 0,
                        to_img, line_width, bounds, scale,
                        text_data, drawn_boxes, None, 0)
        
        # 绘制模型空间实体
        for entity in self.msp:
            try:
                etype = entity.dxftype()
                layer = getattr(entity.dxf, 'layer', '0')
                color = self._get_entity_color(entity, layer)
                
                if etype == 'LINE':
                    sx, sy = entity.dxf.start.x, entity.dxf.start.y
                    ex, ey = entity.dxf.end.x, entity.dxf.end.y
                    
                    if max(sx, ex) < x_min or min(sx, ex) > x_max:
                        continue
                    if max(sy, ey) < y_min or min(sy, ey) > y_max:
                        continue
                    
                    draw.line([to_img(sx, sy), to_img(ex, ey)], fill=color, width=line_width)
                
                elif etype == 'LWPOLYLINE':
                    points = list(entity.get_points('xy'))
                    if not points:
                        continue
                    
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    
                    if max(xs) < x_min or min(xs) > x_max:
                        continue
                    if max(ys) < y_min or min(ys) > y_max:
                        continue
                    
                    img_pts = [to_img(p[0], p[1]) for p in points]
                    if len(img_pts) > 1:
                        draw.line(img_pts, fill=color, width=line_width)
                        if entity.closed and len(img_pts) > 2:
                            draw.line([img_pts[-1], img_pts[0]], fill=color, width=line_width)
                
                elif etype == 'CIRCLE':
                    cx, cy, r = entity.dxf.center.x, entity.dxf.center.y, entity.dxf.radius
                    
                    if cx + r < x_min or cx - r > x_max:
                        continue
                    if cy + r < y_min or cy - r > y_max:
                        continue
                    
                    icx, icy = to_img(cx, cy)
                    ir = int(r * scale)
                    if ir > 0:
                        draw.ellipse([icx - ir, icy - ir, icx + ir, icy + ir],
                                    outline=color, width=line_width)
                
                elif etype == 'ARC':
                    cx, cy, r = entity.dxf.center.x, entity.dxf.center.y, entity.dxf.radius
                    
                    if cx + r < x_min or cx - r > x_max:
                        continue
                    if cy + r < y_min or cy - r > y_max:
                        continue
                    
                    icx, icy = to_img(cx, cy)
                    ir = int(r * scale)
                    if ir > 0:
                        draw.arc([icx - ir, icy - ir, icx + ir, icy + ir],
                                -entity.dxf.end_angle, -entity.dxf.start_angle,
                                fill=color, width=line_width)
                
                elif etype == 'TEXT':
                    tx, ty = entity.dxf.insert.x, entity.dxf.insert.y
                    
                    if tx < x_min - 1000 or tx > x_max + 1000:
                        continue
                    if ty < y_min - 1000 or ty > y_max + 1000:
                        continue
                    
                    text = self._replace_cad_symbols(entity.dxf.text)
                    height = entity.dxf.height
                    rotation = getattr(entity.dxf, 'rotation', 0)
                    
                    if text.strip():
                        text_data.append({
                            'text': text.strip(), 'x': tx, 'y': ty,
                            'height': height, 'layer': layer
                        })
                        self._draw_text_compressed(draw, image, text, tx, ty, height,
                                                  rotation, color, bounds, scale, drawn_boxes)
                
                elif etype == 'MTEXT':
                    tx, ty = entity.dxf.insert.x, entity.dxf.insert.y
                    
                    if tx < x_min - 1000 or tx > x_max + 1000:
                        continue
                    if ty < y_min - 1000 or ty > y_max + 1000:
                        continue
                    
                    text = entity.text
                    text = re.sub(r'\\P', '\n', text)
                    text = re.sub(r'\\[A-Za-z][0-9.]*;?', '', text)
                    text = re.sub(r'[{}]', '', text)
                    text = self._replace_cad_symbols(text)
                    
                    height = getattr(entity.dxf, 'char_height', 2.5)
                    rotation = getattr(entity.dxf, 'rotation', 0)
                    
                    if text.strip():
                        text_data.append({
                            'text': text.strip(), 'x': tx, 'y': ty,
                            'height': height, 'layer': layer
                        })
                        for i, line in enumerate(text.split('\n')):
                            if line.strip():
                                line_y = ty - i * height * 1.5
                                self._draw_text_compressed(draw, image, line, tx, line_y,
                                                          height, rotation, color, bounds,
                                                          scale, drawn_boxes)
                
                elif etype == 'INSERT':
                    block_name = entity.dxf.name
                    if block_name == self.frame_block_name:
                        continue
                    
                    ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
                    isx = getattr(entity.dxf, 'xscale', 1.0)
                    isy = getattr(entity.dxf, 'yscale', 1.0)
                    irot = getattr(entity.dxf, 'rotation', 0)
                    
                    self._draw_block(draw, image, block_name,
                                    ix, iy, isx, isy, irot,
                                    to_img, line_width, bounds, scale,
                                    text_data, drawn_boxes, layer, 0)
                
            except:
                continue
        
        # 保存完整PNG
        full_png_path = os.path.join(frame_dir, f"frame_{frame.index}_full.png")
        image.save(full_png_path, 'PNG', dpi=(OUTPUT_DPI, OUTPUT_DPI))
        print(f"  完整PNG: {full_png_path}")
        
        # 收集文字起始X坐标
        x_starts_unique = self._collect_text_x_starts(x_min, x_max, y_min, y_max)
        print(f"    文字起始X坐标数量: {len(x_starts_unique)}")
        
        # 大模型判断行列数（可选）
        inferred_cols = None
        inferred_rows = None
        inferred_type = "UNKNOWN"
        llm_mask_path = None
        if ENABLE_LLM_GRID:
            llm_mask_path = self._build_llm_mask(full_png_path, frame_dir)
            if llm_mask_path:
                inferred = self._infer_grid_from_llm(llm_mask_path)
                if inferred:
                    inferred_cols, inferred_rows, inferred_type = inferred

        mask_layout = None
        mask_confidence = 0.0
        if llm_mask_path:
            mask_layout, mask_confidence = self._infer_layout_from_mask(llm_mask_path)

        # 手动覆盖（按帧号优先）
        if frame.index in FORCE_STRUCTURED_FRAMES:
            inferred_type = "STRUCTURED"
        elif frame.index in FORCE_TEXT_ONLY_FRAMES:
            inferred_type = "TEXT_ONLY"
        else:
            # 双向校正：LLM 为主，强mask可覆盖
            if mask_layout and mask_confidence >= MASK_CONFIDENCE_THRESHOLD:
                inferred_type = mask_layout
            elif mask_layout and mask_confidence >= MASK_CONFIDENCE_MIN_OVERRIDE:
                inferred_type = mask_layout
            elif inferred_type not in ("TEXT_ONLY", "STRUCTURED"):
                inferred_type = mask_layout if mask_layout else "STRUCTURED"
            else:
                if inferred_type == "STRUCTURED" and mask_layout == "TEXT_ONLY" and mask_confidence >= 0.3:
                    inferred_type = "TEXT_ONLY"
                elif inferred_type == "TEXT_ONLY" and mask_layout == "STRUCTURED" and mask_confidence >= 0.3:
                    inferred_type = "STRUCTURED"

        # 统一使用的目标行列数
        effective_cols = inferred_cols if inferred_cols else TARGET_COLS
        effective_rows = inferred_rows if inferred_rows else TARGET_ROWS
        use_structured_fixed = inferred_type == "STRUCTURED"
        if use_structured_fixed:
            effective_cols = STRUCTURED_FIXED_COLS
            effective_rows = STRUCTURED_FIXED_ROWS

        # 临时覆盖全局切分参数（仅用于本图框切分）
        orig_cols, orig_rows = TARGET_COLS, TARGET_ROWS
        TARGET_COLS = effective_cols
        TARGET_ROWS = effective_rows
        if inferred_cols and inferred_rows:
            print(f"    LLM切分: {TARGET_COLS}列 x {TARGET_ROWS}行 (类型: {inferred_type})")
        if use_structured_fixed:
            print(f"    STRUCTURED 固定切分: {TARGET_COLS}列 x {TARGET_ROWS}行 (使用 fontstest-1 逻辑)")

        # 优先使用 mask 投影定位切分线
        mask_boundaries = None
        mask_cuts_px = None
        if llm_mask_path and not use_structured_fixed:
            mask_boundaries = self._build_boundaries_from_mask(
                llm_mask_path, x_min, x_max, y_min, y_max, scale,
                target_cols=effective_cols, target_rows=effective_rows,
                layout_type=inferred_type
            )
            if mask_boundaries:
                mask_cuts_px = (mask_boundaries[2], mask_boundaries[3])
                print(f"    Mask投影切分: {effective_cols}列 x {effective_rows}行 (策略: {inferred_type})")

        # 计算切分边界
        print(f"    计算切分边界 ({TARGET_COLS}列 x {TARGET_ROWS}行)...")
        if mask_boundaries:
            col_boundaries, row_boundaries_per_col = mask_boundaries[:2]
        else:
            col_boundaries, row_boundaries_per_col = self._calculate_split_boundaries(
                None, x_min, x_max, y_min, y_max, frame,
                cluster_only=use_structured_fixed)
        
        # 打印列边界信息
        col_widths = [col_boundaries[i+1] - col_boundaries[i] for i in range(len(col_boundaries)-1)]
        print(f"    列边界: {[f'{b:.0f}' for b in col_boundaries]}")
        print(f"    列宽度: {[f'{w:.0f}' for w in col_widths]}")
        
        # 绘制调试线条
        if DEBUG_DRAW_LINES:
            print(f"    绘制调试线条...")
            debug_image = image.copy()
            debug_draw = ImageDraw.Draw(debug_image)
            self._draw_debug_lines(debug_draw, debug_image, x_min, x_max, y_min, y_max,
                                  scale, col_boundaries, x_starts_unique,
                                  mask_cuts_px=mask_cuts_px)
            # 保存带调试线条的图片（仅 debug 图）
            debug_png_path = os.path.join(frame_dir, f"frame_{frame.index}_debug.png")
            debug_image.save(debug_png_path, 'PNG', dpi=(OUTPUT_DPI, OUTPUT_DPI))
            print(f"  调试PNG: {debug_png_path}")
        
        # 切分并保存（如果启用）
        if ENABLE_SPLIT:
            split_count = self._split_and_save_image(image, frame_dir, frame.index,
                                                      col_boundaries, row_boundaries_per_col,
                                                      scale, bounds)
            print(f"  切分输出: {split_count} 张图片")
        else:
            print(f"  切分功能已禁用")
        
        # 恢复全局切分参数
        TARGET_COLS, TARGET_ROWS = orig_cols, orig_rows

        # 保存CSV
        csv_path = os.path.join(frame_dir, f"frame_{frame.index}_texts.csv")
        text_data.sort(key=lambda t: (-t['y'], t['x']))
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '文字内容', 'X坐标', 'Y坐标', '字高', '图层'])
            for i, t in enumerate(text_data, 1):
                writer.writerow([i, t['text'], f"{t['x']:.2f}", f"{t['y']:.2f}",
                               f"{t['height']:.2f}", t['layer']])
        print(f"  CSV: {csv_path} ({len(text_data)} 条)")
        
        del image, draw
        gc.collect()
    
    def convert(self):
        print("\n" + "=" * 60)
        print("DXF图框转PNG工具 v13 (支持大块表格切分)")
        print("=" * 60)
        print(f"输入: {INPUT_DXF_PATH}")
        print(f"输出: {OUTPUT_DIR}")
        print(f"背景色: RGB{BACKGROUND_COLOR}")
        print(f"目标切分: {TARGET_COLS}列 x {TARGET_ROWS}行")
        print(f"切分偏移: X方向{CUT_OFFSET_X}mm, Y方向{CUT_OFFSET_Y}mm")
        print(f"输出DPI: {OUTPUT_DPI}")
        
        frames = self.find_frames()
        if not frames:
            print("未找到图框")
            return
        
        print(f"\n找到 {len(frames)} 个图框")
        
        for frame in frames:
            self.render_frame(frame)
            gc.collect()
        
        print("\n" + "=" * 60)
        print("完成！")
        print("=" * 60)


if __name__ == "__main__":
    converter = DXFConverter()
    converter.convert()