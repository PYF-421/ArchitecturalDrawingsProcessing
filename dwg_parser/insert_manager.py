"""
块引用(INSERT)管理模块

负责管理和处理CAD图纸中的块引用(INSERT)实体，包括：
- 块定义(BLOCK_RECORD)的查询和管理
- INSERT实体的查询和过滤
- 嵌套块的展开和坐标变换
- 块属性(ATTRIB)的处理
- 块使用情况统计

CAD块引用结构说明：
- INSERT实体: 块的引用/实例，包含插入点、缩放、旋转等变换信息
- BLOCK_RECORD: 块定义，包含块名、基点和块内实体
- ATTRIB: 块属性值（在INSERT中），对应块定义中的ATTDEF
"""

import math
from typing import Any, Optional, List, Dict, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from dwg_parser.loader import DwgJsonLoader


@dataclass
class BlockDefinition:
    """
    块定义信息
    
    Attributes:
        name: 块名称
        handle: 块句柄
        base_point: 块基点坐标
        entities: 块内实体列表（原始数据）
        insertion_units: 插入单位
        explodability: 是否可分解
        scalability: 是否可缩放
        description: 块描述
        is_anonymous: 是否为匿名块（以*开头）
        is_layout: 是否为布局块（*Paper_Space或*Model_Space）
    """
    name: str
    handle: int
    base_point: Dict[str, float]
    entities: List[dict]
    insertion_units: int = 0
    explodability: bool = True
    scalability: bool = True
    description: str = ""
    is_anonymous: bool = False
    is_layout: bool = False
    
    def __post_init__(self):
        self.is_anonymous = self.name.startswith("*")
        self.is_layout = self.name in ("*Paper_Space", "*Model_Space", "*Paper_Space0")
    
    @property
    def entity_count(self) -> int:
        """块内实体数量"""
        return len(self.entities)
    
    def get_entities_by_type(self, entity_type: str) -> List[dict]:
        """获取块内指定类型的实体"""
        return [e for e in self.entities if e.get("type") == entity_type.upper()]
    
    def has_nested_inserts(self) -> bool:
        """检查块内是否包含嵌套的INSERT"""
        return any(e.get("type") == "INSERT" for e in self.entities)


@dataclass
class InsertInfo:
    """
    INSERT实体信息
    
    Attributes:
        handle: INSERT实体句柄
        block_name: 引用的块名称
        insertion_point: 插入点坐标
        x_scale: X方向缩放
        y_scale: Y方向缩放
        z_scale: Z方向缩放
        rotation: 旋转角度（弧度）
        layer: 所在图层
        column_count: 阵列列数
        row_count: 阵列行数
        column_spacing: 列间距
        row_spacing: 行间距
        attributes: 块属性列表
        raw_data: 原始数据
        owner_block_handle: 所属块的handle
    """
    handle: int
    block_name: str
    insertion_point: Dict[str, float]
    x_scale: float = 1.0
    y_scale: float = 1.0
    z_scale: float = 1.0
    rotation: float = 0.0
    layer: str = "0"
    column_count: int = 0
    row_count: int = 0
    column_spacing: float = 0.0
    row_spacing: float = 0.0
    attributes: List[dict] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)
    owner_block_handle: int = 0
    
    @property
    def is_array(self) -> bool:
        """是否为阵列插入"""
        return self.column_count > 1 or self.row_count > 1
    
    @property
    def has_uniform_scale(self) -> bool:
        """是否为均匀缩放"""
        return abs(self.x_scale - self.y_scale) < 1e-6 and abs(self.y_scale - self.z_scale) < 1e-6
    
    @property
    def is_mirrored(self) -> bool:
        """是否镜像（任一方向缩放为负）"""
        return self.x_scale < 0 or self.y_scale < 0


@dataclass
class TransformedPoint:
    """
    变换后的点坐标
    
    Attributes:
        x: X坐标
        y: Y坐标
        z: Z坐标
        original: 原始坐标
    """
    x: float
    y: float
    z: float = 0.0
    original: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


class InsertManager:
    """
    块引用管理器
    
    管理CAD图纸中的所有块引用(INSERT)实体和块定义(BLOCK_RECORD)
    
    功能：
    - 查询块定义
    - 获取INSERT实体列表
    - 坐标变换（将块内局部坐标转换为世界坐标）
    - 展开嵌套块
    - 处理块属性
    
    Usage:
        from dwg_parser import DwgParser
        
        parser = DwgParser("drawing.json")
        insert_mgr = parser.insert_manager
        
        # 获取所有块定义
        blocks = insert_mgr.get_all_block_definitions()
        
        # 获取指定块的定义
        block = insert_mgr.get_block_definition("MyBlock")
        
        # 获取所有INSERT实体
        inserts = insert_mgr.get_all_inserts()
        
        # 变换坐标
        world_point = insert_mgr.transform_point(local_point, insert_info)
    """
    
    def __init__(self, loader: 'DwgJsonLoader'):
        """
        初始化块引用管理器
        
        Args:
            loader: DwgJsonLoader实例
        """
        self.loader = loader
        
        # 缓存
        self._block_definitions_cache: Optional[Dict[str, BlockDefinition]] = None
        self._block_by_handle_cache: Optional[Dict[int, BlockDefinition]] = None
        self._all_inserts_cache: Optional[List[InsertInfo]] = None
    
    def _clear_cache(self):
        """清除所有缓存"""
        self._block_definitions_cache = None
        self._block_by_handle_cache = None
        self._all_inserts_cache = None
    
    # ============================================================
    # 块定义管理
    # ============================================================
    
    def get_all_block_definitions(self) -> Dict[str, BlockDefinition]:
        """
        获取所有块定义
        
        Returns:
            块名 -> BlockDefinition 的字典
        """
        if self._block_definitions_cache is not None:
            return self._block_definitions_cache
        
        self._block_definitions_cache = {}
        self._block_by_handle_cache = {}
        
        block_records = self.loader.get_block_records()
        
        for block_data in block_records:
            name = block_data.get("name", "")
            if not name:
                continue
            
            block_def = BlockDefinition(
                name=name,
                handle=block_data.get("handle", 0),
                base_point=block_data.get("basePoint", {"x": 0, "y": 0, "z": 0}),
                entities=block_data.get("entities", []),
                insertion_units=block_data.get("insertionUnits", 0),
                explodability=bool(block_data.get("explodability", 1)),
                scalability=bool(block_data.get("scalability", 0)),
                description=block_data.get("description", "")
            )
            
            self._block_definitions_cache[name] = block_def
            self._block_by_handle_cache[block_def.handle] = block_def
        
        return self._block_definitions_cache
    
    def get_block_definition(self, name: str) -> Optional[BlockDefinition]:
        """
        获取指定名称的块定义
        
        Args:
            name: 块名称
            
        Returns:
            BlockDefinition 或 None
        """
        return self.get_all_block_definitions().get(name)
    
    def get_block_definition_by_handle(self, handle: int) -> Optional[BlockDefinition]:
        """
        通过句柄获取块定义
        
        Args:
            handle: 块句柄
            
        Returns:
            BlockDefinition 或 None
        """
        self.get_all_block_definitions()  # 确保缓存已构建
        return self._block_by_handle_cache.get(handle)
    
    def get_user_blocks(self) -> Dict[str, BlockDefinition]:
        """
        获取用户定义的块（排除系统块和匿名块）
        
        Returns:
            用户块定义字典
        """
        all_blocks = self.get_all_block_definitions()
        return {
            name: block for name, block in all_blocks.items()
            if not block.is_anonymous and not block.is_layout
        }
    
    def get_anonymous_blocks(self) -> Dict[str, BlockDefinition]:
        """
        获取匿名块（以*开头的块，如尺寸标注块）
        
        Returns:
            匿名块定义字典
        """
        all_blocks = self.get_all_block_definitions()
        return {
            name: block for name, block in all_blocks.items()
            if block.is_anonymous and not block.is_layout
        }
    
    def block_exists(self, name: str) -> bool:
        """检查块是否存在"""
        return name in self.get_all_block_definitions()
    
    def get_block_names(self) -> List[str]:
        """
        获取所有块名称列表
        
        Returns:
            块名称列表
        """
        return list(self.get_all_block_definitions().keys())
    
    # ============================================================
    # INSERT实体管理
    # ============================================================
    
    def get_all_inserts(self, include_nested: bool = True) -> List[InsertInfo]:
        """
        获取所有INSERT实体
        
        Args:
            include_nested: 是否包含嵌套在块内的INSERT
            
        Returns:
            InsertInfo列表
        """
        if self._all_inserts_cache is not None:
            if include_nested:
                return self._all_inserts_cache
            # 如果不需要嵌套的，过滤只返回Model_Space中的
            return [ins for ins in self._all_inserts_cache if self._is_in_model_space(ins)]
        
        self._all_inserts_cache = []
        block_records = self.loader.get_block_records()
        
        for block_data in block_records:
            block_handle = block_data.get("handle", 0)
            
            for entity in block_data.get("entities", []):
                if entity.get("type") != "INSERT":
                    continue
                
                insert_info = self._parse_insert_entity(entity, block_handle)
                self._all_inserts_cache.append(insert_info)
        
        if include_nested:
            return self._all_inserts_cache
        else:
            return [ins for ins in self._all_inserts_cache if self._is_in_model_space(ins)]
    
    def _is_in_model_space(self, insert: InsertInfo) -> bool:
        """检查INSERT是否直接在Model_Space中"""
        model_space = self.get_block_definition("*Model_Space")
        if model_space and model_space.handle == insert.owner_block_handle:
            return True
        return False
    
    def _parse_insert_entity(self, entity: dict, owner_block_handle: int = 0) -> InsertInfo:
        """
        解析INSERT实体数据
        
        Args:
            entity: INSERT实体原始数据
            owner_block_handle: 所属块的handle
            
        Returns:
            InsertInfo对象
        """
        insertion_point = entity.get("insertionPoint", {})
        
        return InsertInfo(
            handle=entity.get("handle", 0),
            block_name=entity.get("name", ""),
            insertion_point={
                "x": insertion_point.get("x", 0),
                "y": insertion_point.get("y", 0),
                "z": insertion_point.get("z", 0)
            },
            x_scale=entity.get("xScale", 1.0),
            y_scale=entity.get("yScale", 1.0),
            z_scale=entity.get("zScale", 1.0),
            rotation=entity.get("rotation", 0.0),
            layer=entity.get("layer", "0"),
            column_count=entity.get("columnCount", 0),
            row_count=entity.get("rowCount", 0),
            column_spacing=entity.get("columnSpacing", 0.0),
            row_spacing=entity.get("rowSpacing", 0.0),
            attributes=entity.get("attribs", []),
            raw_data=entity,
            owner_block_handle=owner_block_handle
        )
    
    def get_inserts_by_block_name(self, block_name: str) -> List[InsertInfo]:
        """
        获取引用指定块的所有INSERT
        
        Args:
            block_name: 块名称
            
        Returns:
            InsertInfo列表
        """
        return [ins for ins in self.get_all_inserts() if ins.block_name == block_name]
    
    def get_inserts_by_layer(self, layer_name: str) -> List[InsertInfo]:
        """
        获取指定图层上的所有INSERT
        
        Args:
            layer_name: 图层名称
            
        Returns:
            InsertInfo列表
        """
        return [ins for ins in self.get_all_inserts() if ins.layer == layer_name]
    
    def get_inserts_by_layers(self, layer_names: List[str]) -> List[InsertInfo]:
        """
        获取多个图层上的所有INSERT
        
        Args:
            layer_names: 图层名称列表
            
        Returns:
            InsertInfo列表
        """
        layer_set = set(layer_names)
        return [ins for ins in self.get_all_inserts() if ins.layer in layer_set]
    
    # ============================================================
    # 坐标变换
    # ============================================================
    
    def transform_point(
        self,
        local_point: Dict[str, float],
        insert_info: InsertInfo,
        consider_base_point: bool = True
    ) -> TransformedPoint:
        """
        将块内局部坐标变换为世界坐标
        
        变换顺序：
        1. 减去块基点
        2. 应用缩放
        3. 应用旋转
        4. 加上插入点
        
        Args:
            local_point: 局部坐标 {"x": float, "y": float, "z": float}
            insert_info: INSERT信息
            consider_base_point: 是否考虑块基点
            
        Returns:
            TransformedPoint对象
        """
        x = local_point.get("x", 0)
        y = local_point.get("y", 0)
        z = local_point.get("z", 0)
        
        # 1. 减去块基点
        if consider_base_point:
            block_def = self.get_block_definition(insert_info.block_name)
            if block_def:
                base = block_def.base_point
                x -= base.get("x", 0)
                y -= base.get("y", 0)
                z -= base.get("z", 0)
        
        # 2. 应用缩放
        x *= insert_info.x_scale
        y *= insert_info.y_scale
        z *= insert_info.z_scale
        
        # 3. 应用旋转（绕Z轴）
        if abs(insert_info.rotation) > 1e-10:
            cos_r = math.cos(insert_info.rotation)
            sin_r = math.sin(insert_info.rotation)
            new_x = x * cos_r - y * sin_r
            new_y = x * sin_r + y * cos_r
            x, y = new_x, new_y
        
        # 4. 加上插入点
        ins_pt = insert_info.insertion_point
        x += ins_pt.get("x", 0)
        y += ins_pt.get("y", 0)
        z += ins_pt.get("z", 0)
        
        return TransformedPoint(x=x, y=y, z=z, original=local_point)
    
    def transform_points(
        self,
        local_points: List[Dict[str, float]],
        insert_info: InsertInfo
    ) -> List[TransformedPoint]:
        """
        批量变换点坐标
        
        Args:
            local_points: 局部坐标列表
            insert_info: INSERT信息
            
        Returns:
            TransformedPoint列表
        """
        return [self.transform_point(pt, insert_info) for pt in local_points]
    
    def get_transformation_matrix(self, insert_info: InsertInfo) -> List[List[float]]:
        """
        获取INSERT的4x4变换矩阵
        
        返回的矩阵可用于完整的3D变换
        
        Args:
            insert_info: INSERT信息
            
        Returns:
            4x4变换矩阵（行优先）
        """
        cos_r = math.cos(insert_info.rotation)
        sin_r = math.sin(insert_info.rotation)
        sx, sy, sz = insert_info.x_scale, insert_info.y_scale, insert_info.z_scale
        tx = insert_info.insertion_point.get("x", 0)
        ty = insert_info.insertion_point.get("y", 0)
        tz = insert_info.insertion_point.get("z", 0)
        
        # 组合缩放、旋转、平移
        return [
            [sx * cos_r, -sy * sin_r, 0, tx],
            [sx * sin_r,  sy * cos_r, 0, ty],
            [0,           0,          sz, tz],
            [0,           0,          0,  1]
        ]
    
    # ============================================================
    # 块展开
    # ============================================================
    
    def expand_insert(
        self,
        insert_info: InsertInfo,
        max_depth: int = 10,
        transform_coords: bool = True
    ) -> List[dict]:
        """
        展开INSERT，获取块内所有实体（可选变换坐标）
        
        Args:
            insert_info: INSERT信息
            max_depth: 最大嵌套深度
            transform_coords: 是否变换坐标到世界坐标系
            
        Returns:
            展开后的实体列表（带变换后的坐标）
        """
        block_def = self.get_block_definition(insert_info.block_name)
        if not block_def:
            return []
        
        return self._expand_block_recursive(
            block_def, insert_info, max_depth, transform_coords, current_depth=0
        )
    
    def _expand_block_recursive(
        self,
        block_def: BlockDefinition,
        parent_insert: InsertInfo,
        max_depth: int,
        transform_coords: bool,
        current_depth: int
    ) -> List[dict]:
        """递归展开块"""
        if current_depth >= max_depth:
            return []
        
        expanded = []
        
        for entity in block_def.entities:
            entity_type = entity.get("type", "")
            
            if entity_type == "INSERT":
                # 递归展开嵌套的INSERT
                nested_block_name = entity.get("name", "")
                nested_block_def = self.get_block_definition(nested_block_name)
                
                if nested_block_def:
                    nested_insert = self._parse_insert_entity(entity)
                    
                    # 如果需要变换坐标，需要组合变换
                    if transform_coords:
                        # 先变换嵌套INSERT的插入点
                        transformed_pt = self.transform_point(
                            nested_insert.insertion_point, parent_insert
                        )
                        nested_insert.insertion_point = transformed_pt.to_dict()
                        # 组合旋转角度
                        nested_insert.rotation += parent_insert.rotation
                        # 组合缩放
                        nested_insert.x_scale *= parent_insert.x_scale
                        nested_insert.y_scale *= parent_insert.y_scale
                        nested_insert.z_scale *= parent_insert.z_scale
                    
                    nested_entities = self._expand_block_recursive(
                        nested_block_def, nested_insert, max_depth, 
                        transform_coords, current_depth + 1
                    )
                    expanded.extend(nested_entities)
            else:
                # 普通实体，变换坐标
                if transform_coords:
                    transformed_entity = self._transform_entity(entity, parent_insert)
                    expanded.append(transformed_entity)
                else:
                    expanded.append(entity.copy())
        
        return expanded
    
    def _transform_entity(self, entity: dict, insert_info: InsertInfo) -> dict:
        """
        变换单个实体的坐标
        
        根据实体类型，变换其中的坐标字段
        """
        result = entity.copy()
        entity_type = entity.get("type", "")
        
        # 根据实体类型变换相应的坐标字段
        point_fields_map = {
            "LINE": ["startPoint", "endPoint"],
            "CIRCLE": ["center"],
            "ARC": ["center"],
            "TEXT": ["startPoint", "insertionPoint"],
            "MTEXT": ["insertionPoint"],
            "POINT": ["position"],
            "LWPOLYLINE": [],  # 需要特殊处理vertices
            "DIMENSION": ["definitionPoint", "textPoint", "insertionPoint", 
                          "subDefinitionPoint1", "subDefinitionPoint2"],
            "HATCH": [],  # HATCH需要特殊处理
        }
        
        # 变换点字段
        point_fields = point_fields_map.get(entity_type, [])
        for field_name in point_fields:
            if field_name in result and result[field_name]:
                transformed = self.transform_point(result[field_name], insert_info)
                result[field_name] = transformed.to_dict()
        
        # 特殊处理LWPOLYLINE的vertices
        if entity_type == "LWPOLYLINE" and "vertices" in result:
            new_vertices = []
            for v in result["vertices"]:
                pt = {"x": v.get("x", 0), "y": v.get("y", 0), "z": 0}
                transformed = self.transform_point(pt, insert_info)
                new_v = v.copy()
                new_v["x"] = transformed.x
                new_v["y"] = transformed.y
                new_vertices.append(new_v)
            result["vertices"] = new_vertices
        
        # 变换旋转角度（如果有）
        if "rotation" in result and entity_type in ("TEXT", "MTEXT"):
            result["rotation"] = result["rotation"] + insert_info.rotation
        
        return result
    
    # ============================================================
    # 块属性处理
    # ============================================================
    
    def get_insert_attributes(self, insert_info: InsertInfo) -> Dict[str, str]:
        """
        获取INSERT的属性值
        
        Args:
            insert_info: INSERT信息
            
        Returns:
            属性标签 -> 属性值 的字典
        """
        result = {}
        for attr in insert_info.attributes:
            tag = attr.get("tag", "")
            value = attr.get("text", "") or attr.get("value", "")
            if tag:
                result[tag] = value
        return result
    
    def get_block_attribute_definitions(self, block_name: str) -> List[dict]:
        """
        获取块定义中的属性定义(ATTDEF)
        
        Args:
            block_name: 块名称
            
        Returns:
            ATTDEF实体列表
        """
        block_def = self.get_block_definition(block_name)
        if not block_def:
            return []
        
        return block_def.get_entities_by_type("ATTDEF")
    
    # ============================================================
    # 统计和查询
    # ============================================================
    
    def count_inserts_by_block(self) -> Dict[str, int]:
        """
        统计每个块被引用的次数
        
        Returns:
            块名 -> 引用次数 的字典
        """
        counts = {}
        for insert in self.get_all_inserts():
            name = insert.block_name
            counts[name] = counts.get(name, 0) + 1
        return counts
    
    def get_block_usage_info(self) -> List[Dict[str, Any]]:
        """
        获取块使用情况汇总
        
        Returns:
            包含块名、定义、引用次数等信息的列表
        """
        counts = self.count_inserts_by_block()
        result = []
        
        for name, block_def in self.get_all_block_definitions().items():
            result.append({
                "name": name,
                "handle": block_def.handle,
                "is_anonymous": block_def.is_anonymous,
                "is_layout": block_def.is_layout,
                "entity_count": block_def.entity_count,
                "insert_count": counts.get(name, 0),
                "has_nested_inserts": block_def.has_nested_inserts()
            })
        
        return result
    
    def find_unused_blocks(self) -> List[str]:
        """
        查找未被使用的块定义
        
        Returns:
            未被引用的块名列表
        """
        counts = self.count_inserts_by_block()
        all_blocks = self.get_all_block_definitions()
        
        unused = []
        for name, block in all_blocks.items():
            if not block.is_layout and counts.get(name, 0) == 0:
                unused.append(name)
        
        return unused
    
    def get_nesting_depth(self, block_name: str, visited: Optional[set] = None) -> int:
        """
        计算块的最大嵌套深度
        
        Args:
            block_name: 块名称
            visited: 已访问的块（用于检测循环引用）
            
        Returns:
            最大嵌套深度
        """
        if visited is None:
            visited = set()
        
        if block_name in visited:
            return 0  # 循环引用
        
        block_def = self.get_block_definition(block_name)
        if not block_def:
            return 0
        
        visited.add(block_name)
        max_depth = 0
        
        for entity in block_def.entities:
            if entity.get("type") == "INSERT":
                nested_name = entity.get("name", "")
                depth = self.get_nesting_depth(nested_name, visited.copy())
                max_depth = max(max_depth, depth + 1)
        
        return max_depth


__all__ = [
    'InsertManager',
    'InsertInfo',
    'BlockDefinition',
    'TransformedPoint',
]