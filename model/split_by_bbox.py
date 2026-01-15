import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dwg_parser import DwgParser
"""
模块功能：ROI 分割与可渲染 JSON 导出

目标：
1) 从 DWG JSON 中取 Model_Space 顶层实体
2) 顶层 INSERT 通过 insert_mgr.expand_insert(transform_coords=True) 展开为 world 坐标几何实体
3) 用指定“外框 INSERT”展开结果的代表点点云 bbox 作为 ROI
4) 在 ROI 内按百分比框（PctBBox，相对 ROI）进行分区（first_hit 或 all_hits）
5) 导出每个分区为“可渲染 DWG JSON”：
   - 顶层结构完全沿用原始 base JSON（不删字段）
   - 仅替换 entities
   - 仅修复：handle 缺失/重复、layer/color 缺失（不全量重写 handle，尽量不破坏引用链）
6) 支持外框实体不输出（exclude_frame_entities=True）

注: 划分精度取决于图像的划分的一个准确度
"""


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class BBox:
    """轴对齐包围盒（AABB）"""
    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self) -> "BBox":
        return BBox(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
        )

    def contains(self, x: float, y: float) -> bool:
        b = self.normalized()
        return (b.x0 <= x <= b.x1) and (b.y0 <= y <= b.y1)

    @property
    def w(self) -> float:
        b = self.normalized()
        return max(0.0, b.x1 - b.x0)

    @property
    def h(self) -> float:
        b = self.normalized()
        return max(0.0, b.y1 - b.y0)


@dataclass(frozen=True)
class PctBBox:
    """相对 reference bbox 的百分比 bbox"""
    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self) -> "PctBBox":
        return PctBBox(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
        )


# ============================================================
# 核心流程函数（尽量少而集中）
# ============================================================
    """
    Args:
        dwg_json_path (str):
            输入 DWG 的 JSON 文件路径（你的“可渲染模板”）。
            该文件必须包含你前端渲染器所需的顶层结构（例如 header/tables/objects/classes/entities 等）。
            导出时会“尽量保留原顶层结构不变”，仅替换 entities 字段。

        frame_block_name (str):
            外框 INSERT 对应的块名（block name）。
            用于定位外框 INSERT，并通过 expand_insert(transform_coords=True) 展开外框后计算 ROI bbox。
            注意：该参数不用于分割对象本身，只用于确定 ROI 范围。

        frame_handle (int):
            外框 INSERT 的 handle（用于唯一锁定该外框实例）。
            同名 block 通常有多个 INSERT，handle 用来指定你要的那一个。

        pct_boxes (List[Tuple[str, PctBBox]]):
            ROI 内部的“分区定义”，使用百分比 bbox（PctBBox）描述相对位置。
            结构为 [(box_id, PctBBox), ...]：
              - box_id: 分区标识字符串（用于输出文件名，例如 out_prefix_{box_id}.json）
              - PctBBox: 相对 ROI bbox 的比例框，范围通常为 [0,1]：
                    x = roi.x0 + pct_x * roi.w
                    y = roi.y0 + pct_y * roi.h
            示例：把 ROI 按纵向切成 3 段：
                ("box1", PctBBox(0.0, 0.0, 1.0, 0.3))
                ("box2", PctBBox(0.0, 0.3, 1.0, 0.6))
                ("box3", PctBBox(0.0, 0.6, 1.0, 1.0))

        out_dir (str):
            输出目录路径。函数会自动创建该目录（os.makedirs(exist_ok=True)）。
            每个分区导出的可渲染 JSON 都会写到此目录。

        out_prefix (str):
            输出文件名前缀。
            输出文件一般形如：
                {out_dir}/{out_prefix}_{box_id}.json
                {out_dir}/{out_prefix}_others.json（如果 include_others=True）
                {out_dir}/{out_prefix}_out_of_roi.json（用于调试 ROI 外内容）

        max_depth (int, default=10):
            展开 INSERT 的最大嵌套深度。
            用于：
              - 扁平化 Model_Space 顶层 INSERT：expand_insert(..., max_depth=max_depth)
              - 展开外框 INSERT：expand_insert(..., max_depth=max_depth)
            若图纸块嵌套很深，可以增大；过大可能导致展开量过大、速度变慢。

        assign_strategy (str, default="first_hit"):
            ROI 内分区时的“归属策略”，决定一个实体是否允许落入多个分区：
              - "first_hit": 若代表点命中多个分区，只归入第一个命中的 box（互斥分配）
              - "all_hits": 若代表点命中多个分区，同时加入所有命中的 box（多归属）
            典型场景：
              - 分区不重叠（推荐 first_hit）：更干净，避免重复实体
              - 分区可能重叠或希望保留交叠部分（用 all_hits）

        exclude_frame_entities (bool, default=True):
            是否将“外框自身的实体”从输出里剔除。
            常见需求是“外框只是 ROI 边界，不希望出现在分区结果中”，因此默认 True。
            实现方式通常是：收集 frame_entities 的 handle 集合，然后从 world_entities 中过滤掉这些 handle。

        include_others (bool, default=True):
            是否导出一个 others 分区：
              - others = ROI 内但未命中任何 pct_boxes 的实体（或代表点缺失导致无法命中）
            若为 True，会额外生成：
                {out_prefix}_others.json
        default_layer (str, default="0"):
            导出时的字段补全默认值：
            若某些实体缺失 layer（或为空），则写入该默认 layer。
            用于增强前端渲染器的健壮性（避免读取字段时报错）。

        default_color (int, default=7):
            导出时的字段补全默认值：
            若某些实体缺失 color（或为 None），则写入该默认 color。
            7 在很多 CAD 语义里表示“白/黑（随背景）”，用于最保底的可见性。

    Returns:
        Dict[str, Any]:
            返回运行汇总信息（summary），通常包括：
              - roi_bbox: 计算得到的 ROI 边界
              - counts: 扁平化数量、ROI 内外数量
              - boxes_abs: pct_boxes 映射到绝对坐标后的 bbox（便于核对）
              - outputs: 实际导出的文件路径列表
              - assign_strategy: 本次使用的归属策略
    """
def split_and_export_by_frame_insert(
    dwg_json_path: str,
    *,
    frame_block_name: str,
    frame_handle: int,
    pct_boxes: List[Tuple[str, PctBBox]],
    out_dir: str,
    out_prefix: str,
    max_depth: int = 10,
    assign_strategy: str = "first_hit",  # "first_hit" or "all_hits"
    exclude_frame_entities: bool = True,
    include_others: bool = True,
    default_layer: str = "0",
    default_color: int = 7,
) -> Dict[str, Any]:
    """
    策略：
    1) 取 Model_Space 顶层实体（EntityManager）
    2) 顶层 INSERT -> expand_insert(transform_coords=True) 展开到 world
    3) 用外框 INSERT 展开后的实体点云 bbox 作为 ROI
    4) (可选) 外框实体不输出
    5) 在 ROI 内按 pct_boxes 分区
    6) 每个分区导出为“可渲染 JSON”（顶层完全保留 base，只替换 entities）
       且不全量重写 handle（只修复缺失/重复），避免破坏引用链导致渲染失败

    Returns:
        summary: 统计信息与 ROI bbox / 输出目录等
    """

    # ----------------------------
    # 0) 读取 base JSON（渲染模板）
    # ----------------------------
    with open(dwg_json_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    parser = DwgParser(dwg_json_path)
    insert_mgr = parser.insert_manager
    entity_mgr = parser.entity_manager

    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # 1) 建 INSERT 索引（Model_Space 顶层）
    # ----------------------------
    # 不做任何坐标变换逻辑，只是索引
    insert_index: Dict[int, Any] = {}
    for ins in insert_mgr.get_all_inserts(include_nested=False):
        try:
            insert_index[int(ins.handle)] = ins
        except Exception:
            # handle 如果不是纯 int，这里就跳过（也可扩展十六进制解析）
            continue

    # ----------------------------
    # 2) Model_Space 顶层 -> world 扁平化
    # ----------------------------
    top_level_infos = entity_mgr.get_all_entities(include_nested=False)

    world_entities: List[Dict[str, Any]] = []
    missed_insert = 0

    for ei in top_level_infos:
        raw = ei.raw_data
        et = (raw.get("type") or "").upper()

        if et != "INSERT":
            # 顶层非 INSERT：通常已经是 world（Model_Space）
            world_entities.append(raw)
            continue

        # 顶层 INSERT：用库 expand_insert 展开 + 转 world
        ins = insert_index.get(int(ei.handle))
        if ins is None:
            missed_insert += 1
            continue

        expanded = insert_mgr.expand_insert(
            ins,
            max_depth=max_depth,
            transform_coords=True
        )
        world_entities.extend(expanded)

    if missed_insert:
        print(f"[WARN] missed top-level INSERT due to index not found: {missed_insert}")

    print(f"[INFO] World flattened entities: {len(world_entities)}")

    # ----------------------------
    # 3) 找外框 INSERT，并算 ROI bbox
    # ----------------------------
    frame_inserts = insert_mgr.get_inserts_by_block_name(frame_block_name)
    target_frame = None
    for ins in frame_inserts:
        if int(ins.handle) == int(frame_handle):
            target_frame = ins
            break
    if target_frame is None:
        raise RuntimeError(f"外框 INSERT 未找到：block={frame_block_name}, handle={frame_handle}")

    frame_entities = insert_mgr.expand_insert(
        target_frame,
        max_depth=max_depth,
        transform_coords=True
    )

    # 用“代表点点云 bbox”作为 ROI（与你当前策略一致）
    roi_bbox = _entities_points_bbox(frame_entities)
    if roi_bbox is None:
        raise RuntimeError("无法计算 ROI bbox：外框展开后没有可用代表点。")

    roi_bbox = roi_bbox.normalized()
    print(f"[INFO] ROI bbox: {roi_bbox}")

    # ----------------------------
    # 4) 外框不输出：剔除外框实体（按 handle）
    # ----------------------------
    if exclude_frame_entities:
        banned = _build_handle_set(frame_entities)
        before = len(world_entities)
        world_entities = [e for e in world_entities if not _handle_in_set(e.get("handle"), banned)]
        print(f"[INFO] Exclude frame entities: {before} -> {len(world_entities)}")

    # ----------------------------
    # 5) ROI 过滤：按代表点落入 ROI
    # ----------------------------
    in_roi, out_roi = [], []
    for e in world_entities:
        p = _entity_anchor_point(e)
        if p is None:
            out_roi.append(e)
            continue
        if roi_bbox.contains(p[0], p[1]):
            in_roi.append(e)
        else:
            out_roi.append(e)

    print(f"[INFO] Entities in ROI: {len(in_roi)}, out of ROI: {len(out_roi)}")

    # ----------------------------
    # 6) ROI 内分区（PctBBox 相对 ROI）
    # ----------------------------
    # 先把 pct_boxes 转绝对 bbox
    abs_boxes: List[Tuple[str, BBox]] = []
    for bid, pb in pct_boxes:
        abs_boxes.append((bid, _pct_to_abs_bbox(pb, roi_bbox)))

    # 分配：first_hit 或 all_hits
    # first_hit 代表唯一归属，每一个点只会出现在一个框内
    # all_hits 代表多归属，每一个点可以出现在多个框内
    box_entities: Dict[str, List[Dict[str, Any]]] = {bid: [] for bid, _ in abs_boxes}
    others: List[Dict[str, Any]] = []

    for e in in_roi:
        p = _entity_anchor_point(e)
        if p is None:
            others.append(e)
            continue

        hits = []
        for bid, bb in abs_boxes:
            if bb.contains(p[0], p[1]):
                hits.append(bid)

        if not hits:
            others.append(e)
            continue

        if assign_strategy == "first_hit":
            box_entities[hits[0]].append(e)
        elif assign_strategy == "all_hits":
            for bid in hits:
                box_entities[bid].append(e)
        else:
            raise ValueError("assign_strategy 必须是 'first_hit' 或 'all_hits'。")

    # ----------------------------
    # 7) 导出：保持 base 顶层结构完全不动，只替换 entities
    #    并且只做“缺失/重复 handle 修复”，避免破坏引用链
    #    为了防止渲染出问题
    # ----------------------------
    outputs: List[str] = []

    for bid, ents in box_entities.items():
        ents_fixed = _fix_entities_for_render(ents, base, default_layer, default_color)
        out_json = dict(base)
        out_json["entities"] = ents_fixed

        out_path = os.path.join(out_dir, f"{out_prefix}_{bid}.json")
        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(out_json, wf, ensure_ascii=False, indent=2)
        outputs.append(out_path)
        print(f"[OK] {out_path}  entities={len(ents_fixed)}")

    if include_others:
        ents_fixed = _fix_entities_for_render(others, base, default_layer, default_color)
        out_json = dict(base)
        out_json["entities"] = ents_fixed

        out_path = os.path.join(out_dir, f"{out_prefix}_others.json")
        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(out_json, wf, ensure_ascii=False, indent=2)
        outputs.append(out_path)
        print(f"[OK] {out_path}  entities={len(ents_fixed)}")

    # 可选：输出 ROI 外（调试）
    out_roi_path = os.path.join(out_dir, f"{out_prefix}_out_of_roi.json")
    out_roi_fixed = _fix_entities_for_render(out_roi, base, default_layer, default_color)
    out_roi_json = dict(base)
    out_roi_json["entities"] = out_roi_fixed
    with open(out_roi_path, "w", encoding="utf-8") as wf:
        json.dump(out_roi_json, wf, ensure_ascii=False, indent=2)
    outputs.append(out_roi_path)
    print(f"[OK] {out_roi_path}  entities={len(out_roi_fixed)}")

    return {
        "dwg_json_path": dwg_json_path,
        "roi_bbox": {"x0": roi_bbox.x0, "y0": roi_bbox.y0, "x1": roi_bbox.x1, "y1": roi_bbox.y1},
        "counts": {
            "world_flattened": len(world_entities),
            "in_roi": len(in_roi),
            "out_roi": len(out_roi),
        },
        "boxes_abs": {bid: {"x0": bb.x0, "y0": bb.y0, "x1": bb.x1, "y1": bb.y1} for bid, bb in abs_boxes},
        "outputs": outputs,
    }


# ============================================================
# 内部工具（集中在文件底部，避免函数碎散）
# 感觉只有在这个文件当中会使用到
# ============================================================

def _pt(d: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(d, dict):
        return None
    return float(d.get("x", 0.0)), float(d.get("y", 0.0))


def _entity_anchor_point(e: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    代表点策略（按点判定）：
      LINE: startPoint
      CIRCLE/ARC: center
      TEXT/MTEXT: insertionPoint or startPoint
      POINT: position
      LWPOLYLINE: vertices[0]
      DIMENSION: textPoint > insertionPoint > definitionPoint
      fallback: start/end/center/insertion/position
    """
    et = (e.get("type") or "").upper()

    if et == "LINE":
        return _pt(e.get("startPoint"))
    if et in ("CIRCLE", "ARC"):
        return _pt(e.get("center"))
    if et in ("TEXT", "MTEXT"):
        return _pt(e.get("insertionPoint") or e.get("startPoint"))
    if et == "POINT":
        return _pt(e.get("position"))
    if et == "LWPOLYLINE":
        vs = e.get("vertices") or []
        if vs:
            v0 = vs[0]
            return float(v0.get("x", 0.0)), float(v0.get("y", 0.0))
        return None
    if et == "DIMENSION":
        for k in ("textPoint", "insertionPoint", "definitionPoint"):
            if e.get(k):
                return _pt(e.get(k))
        return None

    for k in ("startPoint", "endPoint", "center", "insertionPoint", "position"):
        if e.get(k):
            return _pt(e.get(k))
    return None


def _entities_points_bbox(entities: List[Dict[str, Any]]) -> Optional[BBox]:
    """
    根据一组实体的“代表点（anchor point）”计算整体的包围盒 BBox。

    设计目的：
    - 不使用实体几何外轮廓（bbox），而是用代表点做近似统计
    - 适合以下场景：
        * ROI 外框计算
        * INSERT 展开后粗粒度范围估计
        * 分区（pct box）时作为参考 bbox

    注意：
    - 依赖 _entity_anchor_point(e) 提供 world 坐标下的代表点
    - 如果实体没有可用代表点（如异常实体），会被自动跳过
    - 若所有实体都无法提供代表点，返回 None

    Args:
        entities: 实体 raw_data 列表（dict）

    Returns:
        BBox: 覆盖所有代表点的最小轴对齐包围盒
        None: 当没有任何有效代表点时
    """
    xs, ys = [], []
    for e in entities:
        p = _entity_anchor_point(e)
        if p is None:
            continue
        xs.append(p[0])
        ys.append(p[1])
    if not xs:
        return None
    return BBox(min(xs), min(ys), max(xs), max(ys))



def _pct_to_abs_bbox(pb: PctBBox, ref: BBox) -> BBox:
    """
    将“百分比 bbox（PctBBox）”转换为绝对坐标系下的 BBox。

    使用场景：
    - ROI 内部的逻辑分区（如：上/中/下区域）
    - pct box 的坐标是相对于 ref bbox 的比例（0~1）

    转换规则：
        abs_x = ref.x0 + pct_x * ref.width
        abs_y = ref.y0 + pct_y * ref.height

    说明：
    - pb 会先 normalized（确保 x0<=x1, y0<=y1）
    - ref 也会 normalized，避免传入反向 bbox
    - 最终结果也是 normalized 的 BBox

    Args:
        pb: 百分比 bbox（相对 ref）
        ref: 参考的绝对 bbox（如 ROI bbox）

    Returns:
        转换后的绝对坐标 BBox
    """
    pb = pb.normalized()
    ref = ref.normalized()
    return BBox(
        ref.x0 + pb.x0 * ref.w,
        ref.y0 + pb.y0 * ref.h,
        ref.x0 + pb.x1 * ref.w,
        ref.y0 + pb.y1 * ref.h,
    ).normalized()



def _parse_handle_int(h: Any) -> Optional[int]:
    """
    将 JSON 中各种形式的 handle 尽可能解析为 int。
    这一点只是为了安全考虑，导出的handle目前基本上是int了
    Args:
        h: handle 原始值（Any）

    Returns:
        int: 解析成功的 handle
        None: 无法解析或非法
    """
    if h is None:
        return None
    if isinstance(h, int):
        return h
    if isinstance(h, str):
        s = h.strip()
        if not s:
            return None
        # 0xABCD 形式
        if s.lower().startswith("0x"):
            try:
                return int(s, 16)
            except Exception:
                return None
        # 十进制字符串
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
        # 无前缀的十六进制字符串
        try:
            return int(s, 16)
        except Exception:
            return None
    return None


def _build_handle_set(entities: List[Dict[str, Any]]) -> set[int]:
    """
    从实体列表中构建一个“handle 的 int 集合”，防止冲突

    使用场景：
    - 外框（frame）实体排除
    - 重复实体过滤
    - handle 冲突检测

    特点：
    - 使用 _parse_handle_int 做统一解析
    - 自动忽略无法解析 handle 的实体
    - 结果只包含 int，方便 O(1) 查询

    Args:
        entities: 实体 raw_data 列表

    Returns:
        set[int]: 实体 handle 的整数集合
    """
    s: set[int] = set()
    for e in entities:
        hi = _parse_handle_int(e.get("handle"))
        if hi is not None:
            s.add(hi)
    return s



def _handle_in_set(h: Any, banned: set[int]) -> bool:
    """
    判断一个 handle 是否在“禁止集合（banned handles）”中。

    设计目的：
    - 提供一个统一的、安全的 handle 判断入口
    - 避免在主逻辑中反复写解析代码

    使用场景：
    - 判断某实体是否属于“外框实体”
    - 判断是否需要剔除该实体

    Args:
        h: 实体的 handle（原始值）
        banned: 已解析为 int 的 handle 集合

    Returns:
        True: handle 可解析且在 banned 集合中
        False: handle 不在 banned 集合或无法解析
    """
    hi = _parse_handle_int(h)
    return (hi is not None) and (hi in banned)



def _fix_entities_for_render(
    entities: List[Dict[str, Any]],
    base: Dict[str, Any],
    default_layer: str,
    default_color: int,
) -> List[Dict[str, Any]]:
    """
    仅做渲染必要修复：
      - handle 缺失/重复：分配新的唯一 handle（不全量重写）
      - layer/color 缺失：补齐字段

    重要：不全量重写 handle，避免破坏 objects/tables 引用链。
    """
    # 收集 base 中已有 handle，避免冲突
    used: set[int] = set()

    def walk(obj: Any):
        if isinstance(obj, dict):
            if "handle" in obj:
                hi = _parse_handle_int(obj.get("handle"))
                if hi is not None:
                    used.add(hi)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(base)

    next_handle = (max(used) + 1) if used else 1000000

    out: List[Dict[str, Any]] = []
    seen_local: set[int] = set()

    for e in entities:
        ee = dict(e)

        hi = _parse_handle_int(ee.get("handle"))
        if hi is None or hi in seen_local:
            # 分配新 handle（输出类型尽量保持为原来风格：原来是 str 就输出 hex str）
            if isinstance(ee.get("handle"), str):
                ee["handle"] = format(next_handle, "X")
            else:
                ee["handle"] = next_handle
            hi = next_handle
            next_handle += 1

        seen_local.add(hi)

        if not ee.get("layer"):
            ee["layer"] = default_layer
        if "color" not in ee or ee["color"] is None:
            ee["color"] = default_color

        out.append(ee)

    return out


# ============================================================
# 示例调用
# ============================================================

if __name__ == "__main__":
    summary = split_and_export_by_frame_insert(
        dwg_json_path="../(6审)1#_t3.dwg (1).json",
        frame_block_name="民用院-A1S",
        frame_handle=95883,
        pct_boxes=[
            ("pct_box_01", PctBBox(0.00, 0.000, 1.00, 0.229)),
            ("pct_box_02", PctBBox(0.00, 0.229, 1.00, 0.591)),
            ("pct_box_03", PctBBox(0.00, 0.591, 1.00, 0.950)),
        ],
        out_dir="out_renderable_split",
        out_prefix="ROI_A1S_95883",  # 导出群文件前缀
        max_depth=10,
        assign_strategy="first_hit",
        exclude_frame_entities=True,
        include_others=True,
        default_layer="0",
        default_color=7,
    )
    print("DONE summary:", summary)
