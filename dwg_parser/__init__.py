"""
DWG JSON Parser - CAD建筑图纸解析工具包

支持从文件路径、字典或字节数据加载JSON
"""

from typing import Any, Optional, Union

import json
import re
import math
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ============================================================
# loader.py
# ============================================================

class DwgJsonLoader:
    """
    DWG JSON加载器
    
    支持三种初始化方式：
    1. 从文件路径加载: DwgJsonLoader(json_path="path/to/file.json")
    2. 从字典加载: DwgJsonLoader(json_data={"tables": {...}})
    3. 从字节数据加载: DwgJsonLoader(json_bytes=b'{"tables": {...}}')
    """
    
    def __init__(
        self, 
        json_path: Optional[str] = None,
        json_data: Optional[dict] = None,
        json_bytes: Optional[bytes] = None
    ):
        self.json_path = json_path
        self._raw_data: Optional[dict] = None
        self._block_records_cache: Optional[list] = None
        self._block_map_cache: Optional[dict] = None
        
        if json_data is not None:
            self._raw_data = json_data
        elif json_bytes is not None:
            self._raw_data = json.loads(json_bytes.decode('utf-8'))
    
    @classmethod
    def from_file(cls, json_path: str) -> 'DwgJsonLoader':
        loader = cls(json_path=json_path)
        loader.load()
        return loader
    
    @classmethod
    def from_dict(cls, json_data: dict) -> 'DwgJsonLoader':
        return cls(json_data=json_data)
    
    @classmethod
    def from_bytes(cls, json_bytes: bytes) -> 'DwgJsonLoader':
        return cls(json_bytes=json_bytes)
    
    @classmethod
    def from_response(cls, response) -> 'DwgJsonLoader':
        return cls(json_bytes=response.content)
    
    def load(self) -> dict:
        if self._raw_data is not None:
            return self._raw_data
        if self.json_path is None:
            raise ValueError("未指定JSON文件路径，且未提供数据")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self._raw_data = json.load(f)
        return self._raw_data
    
    def load_from_dict(self, json_data: dict) -> dict:
        self._raw_data = json_data
        self._clear_cache()
        return self._raw_data
    
    def load_from_bytes(self, json_bytes: bytes) -> dict:
        self._raw_data = json.loads(json_bytes.decode('utf-8'))
        self._clear_cache()
        return self._raw_data
    
    def _clear_cache(self):
        self._block_records_cache = None
        self._block_map_cache = None
    
    def get_raw_data(self) -> dict:
        if self._raw_data is None:
            self.load()
        return self._raw_data
    
    def get_tables(self) -> dict:
        return self.get_raw_data().get("tables", {})
    
    def get_block_records(self) -> list:
        if self._block_records_cache is None:
            tables = self.get_tables()
            block_record = tables.get("BLOCK_RECORD", {})
            self._block_records_cache = block_record.get("entries", [])
        return self._block_records_cache
    
    def get_block_map(self) -> dict:
        if self._block_map_cache is None:
            self._block_map_cache = {}
            for block in self.get_block_records():
                name = block.get("name", "")
                if name:
                    self._block_map_cache[name] = block
                handle = block.get("handle")
                if handle is not None:
                    self._block_map_cache[handle] = block
        return self._block_map_cache
    
    def get_block_by_name(self, name: str) -> Optional[dict]:
        return self.get_block_map().get(name)
    
    def get_block_by_handle(self, handle: int) -> Optional[dict]:
        return self.get_block_map().get(handle)
    
    def get_layers(self) -> list:
        tables = self.get_tables()
        layer_table = tables.get("LAYER", {})
        return layer_table.get("entries", [])
    
    def get_entities(self) -> list:
        return self.get_raw_data().get("entities", [])


# ============================================================
# entity_manager.py
# ============================================================

@dataclass
class EntityInfo:
    raw_data: dict
    entity_type: str
    handle: int
    layer: str
    block_path: list = field(default_factory=list)
    owner_block: str = ""
    
    def __post_init__(self):
        if self.block_path:
            self.owner_block = self.block_path[-1]
    
    def is_in_model_space(self) -> bool:
        return len(self.block_path) == 1 and self.block_path[0] == "*Model_Space"
    
    def get_nesting_depth(self) -> int:
        return len(self.block_path) - 1


class EntityManager:
    def __init__(self, loader: DwgJsonLoader):
        self.loader = loader
        self._all_entities_cache: Optional[list] = None
    
    def _extract_entities_from_block(self, block: dict, current_path: list) -> list:
        entities = []
        block_name = block.get("name", "")
        new_path = current_path + [block_name]
        
        for entity_data in block.get("entities", []):
            entity_type = entity_data.get("type", "")
            entity_info = EntityInfo(
                raw_data=entity_data,
                entity_type=entity_type,
                handle=entity_data.get("handle", 0),
                layer=entity_data.get("layer", ""),
                block_path=new_path.copy()
            )
            entities.append(entity_info)
            
            if entity_type == "INSERT":
                insert_block_name = entity_data.get("name", "")
                if insert_block_name:
                    referenced_block = self.loader.get_block_by_name(insert_block_name)
                    if referenced_block:
                        nested_entities = self._extract_entities_from_block(referenced_block, new_path)
                        entities.extend(nested_entities)
        return entities
    
    def get_all_entities(self, include_nested: bool = True) -> list:
        if self._all_entities_cache is not None:
            if include_nested:
                return self._all_entities_cache
            else:
                return [e for e in self._all_entities_cache if e.is_in_model_space()]
        
        entities = []
        model_space = self.loader.get_block_by_name("*Model_Space")
        if model_space:
            entities = self._extract_entities_from_block(model_space, [])
        self._all_entities_cache = entities
        
        if include_nested:
            return entities
        else:
            return [e for e in entities if e.is_in_model_space()]
    
    def get_entities_by_type(self, entity_type: str) -> list:
        return [e for e in self.get_all_entities() if e.entity_type == entity_type.upper()]
    
    def get_entities_by_layer(self, layer_names: list) -> list:
        layer_set = set(layer_names)
        return [e for e in self.get_all_entities() if e.layer in layer_set]
    
    def get_entities_by_type_and_layer(self, entity_type: str, layer_names: list) -> list:
        layer_set = set(layer_names)
        return [e for e in self.get_all_entities() if e.entity_type == entity_type.upper() and e.layer in layer_set]
    
    def get_entity_types(self) -> list:
        return list(set(e.entity_type for e in self.get_all_entities()))
    
    def count_entities_by_type(self) -> dict:
        counts = {}
        for e in self.get_all_entities():
            counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
        return counts


# ============================================================
# layer_manager.py
# ============================================================

class LayerManager:
    def __init__(self, loader: DwgJsonLoader):
        self.loader = loader
        self._layer_names_cache: Optional[list] = None
        self._layer_map_cache: Optional[dict] = None
    
    def get_all_layer_names(self) -> list:
        if self._layer_names_cache is None:
            layers = self.loader.get_layers()
            self._layer_names_cache = [layer.get("name", "") for layer in layers if layer.get("name", "")]
        return self._layer_names_cache
    
    def get_layer_map(self) -> dict:
        if self._layer_map_cache is None:
            layers = self.loader.get_layers()
            self._layer_map_cache = {layer.get("name", ""): layer for layer in layers if layer.get("name", "")}
        return self._layer_map_cache
    
    def match_layers(self, pattern: str) -> list:
        compiled_pattern = re.compile(pattern)
        return [name for name in self.get_all_layer_names() if compiled_pattern.search(name)]
    
    def match_layers_fullmatch(self, pattern: str) -> list:
        compiled_pattern = re.compile(pattern)
        return [name for name in self.get_all_layer_names() if compiled_pattern.fullmatch(name)]
    
    def get_layer_info(self, layer_name: str) -> Optional[dict]:
        return self.get_layer_map().get(layer_name)
    
    def layer_exists(self, layer_name: str) -> bool:
        return layer_name in self.get_layer_map()


# ============================================================
# utils.py
# ============================================================

def calculate_line_angle(start: dict, end: dict) -> float:
    dx = end.get("x", 0) - start.get("x", 0)
    dy = end.get("y", 0) - start.get("y", 0)
    return math.atan2(dy, dx)

def calculate_line_length(start: dict, end: dict) -> float:
    dx = end.get("x", 0) - start.get("x", 0)
    dy = end.get("y", 0) - start.get("y", 0)
    dz = end.get("z", 0) - start.get("z", 0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def calculate_arc_endpoint(center: dict, radius: float, angle: float) -> dict:
    return {"x": center.get("x", 0) + radius * math.cos(angle), "y": center.get("y", 0) + radius * math.sin(angle), "z": center.get("z", 0)}

def calculate_arc_endpoints(center: dict, radius: float, start_angle: float, end_angle: float) -> tuple:
    return calculate_arc_endpoint(center, radius, start_angle), calculate_arc_endpoint(center, radius, end_angle)

def calculate_arc_total_angle(start_angle: float, end_angle: float) -> float:
    total = end_angle - start_angle
    return total + 2 * math.pi if total < 0 else total

def calculate_arc_length(radius: float, start_angle: float, end_angle: float) -> float:
    return radius * calculate_arc_total_angle(start_angle, end_angle)

def is_polyline_closed(flag: int) -> bool:
    return bool(flag & 1)

def calculate_bounding_box_line(start: dict, end: dict) -> dict:
    return {"min": {"x": min(start.get("x", 0), end.get("x", 0)), "y": min(start.get("y", 0), end.get("y", 0))},
            "max": {"x": max(start.get("x", 0), end.get("x", 0)), "y": max(start.get("y", 0), end.get("y", 0))}}

def calculate_bounding_box_circle(center: dict, radius: float) -> dict:
    cx, cy = center.get("x", 0), center.get("y", 0)
    return {"min": {"x": cx - radius, "y": cy - radius}, "max": {"x": cx + radius, "y": cy + radius}}

def calculate_bounding_box_points(points: list) -> dict:
    if not points:
        return {"min": {"x": 0, "y": 0}, "max": {"x": 0, "y": 0}}
    xs = [p.get("x", 0) for p in points]
    ys = [p.get("y", 0) for p in points]
    return {"min": {"x": min(xs), "y": min(ys)}, "max": {"x": max(xs), "y": max(ys)}}

def point_to_dict(point: Optional[dict]) -> dict:
    if point is None:
        return {"x": 0, "y": 0, "z": 0}
    return {"x": point.get("x", 0), "y": point.get("y", 0), "z": point.get("z", 0)}

def calculate_bounding_box_arc(center: dict, radius: float, start_angle: float, end_angle: float) -> dict:
    cx, cy = center.get("x", 0), center.get("y", 0)
    start_x, start_y = cx + radius * math.cos(start_angle), cy + radius * math.sin(start_angle)
    end_x, end_y = cx + radius * math.cos(end_angle), cy + radius * math.sin(end_angle)
    min_x, max_x = min(start_x, end_x), max(start_x, end_x)
    min_y, max_y = min(start_y, end_y), max(start_y, end_y)
    
    def normalize(a):
        while a < 0: a += 2 * math.pi
        while a >= 2 * math.pi: a -= 2 * math.pi
        return a
    sa, ea = normalize(start_angle), normalize(end_angle)
    def in_arc(a):
        a = normalize(a)
        return (sa <= a <= ea) if sa <= ea else (a >= sa or a <= ea)
    if in_arc(0): max_x = cx + radius
    if in_arc(math.pi/2): max_y = cy + radius
    if in_arc(math.pi): min_x = cx - radius
    if in_arc(3*math.pi/2): min_y = cy - radius
    return {"min": {"x": min_x, "y": min_y}, "max": {"x": max_x, "y": max_y}}


# ============================================================
# 实体提取器
# ============================================================

class BaseEntityExtractor(ABC):
    ENTITY_TYPE: str = ""
    
    @abstractmethod
    def extract(self, entity_info: Any) -> dict:
        pass
    
    def _extract_common_attrs(self, entity_info: Any) -> dict:
        raw = entity_info.raw_data
        return {
            "layer": entity_info.layer, "block_path": entity_info.block_path.copy(),
            "owner_block": entity_info.owner_block, "color": raw.get("colorIndex", 256),
            "color_name": raw.get("colorName", ""), "handle": entity_info.handle,
            "is_visible": raw.get("isVisible", True), "line_type": raw.get("lineType", ""),
            "line_weight": raw.get("lineweight", 0),
        }


class TextExtractor(BaseEntityExtractor):
    ENTITY_TYPE = "TEXT"
    def extract(self, entity_info: Any) -> dict:
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        position = point_to_dict(raw.get("startPoint"))
        height, text_content = raw.get("textHeight", 0), raw.get("text", "")
        rotation, x_scale = raw.get("rotation", 0), raw.get("xScale", 1.0)
        text_width = len(text_content) * height * 0.6 * x_scale
        px, py = position.get("x", 0), position.get("y", 0)
        if rotation == 0:
            bbox = {"min": {"x": px, "y": py}, "max": {"x": px + text_width, "y": py + height}}
        else:
            corners = [(0, 0), (text_width, 0), (text_width, height), (0, height)]
            cos_r, sin_r = math.cos(rotation), math.sin(rotation)
            xs = [px + dx * cos_r - dy * sin_r for dx, dy in corners]
            ys = [py + dx * sin_r + dy * cos_r for dx, dy in corners]
            bbox = {"min": {"x": min(xs), "y": min(ys)}, "max": {"x": max(xs), "y": max(ys)}}
        return {**common, "position": position, "text_content": text_content, "height": height,
                "rotation": rotation, "x_scale": x_scale, "style_name": raw.get("styleName", ""),
                "oblique_angle": raw.get("obliqueAngle", 0), "h_align": raw.get("halign", 0),
                "v_align": raw.get("valign", 0), "bounding_box": bbox}


class MTextExtractor(BaseEntityExtractor):
    ENTITY_TYPE = "MTEXT"
    def extract(self, entity_info: Any) -> dict:
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        position = point_to_dict(raw.get("insertionPoint"))
        height, text_content = raw.get("textHeight", 0), raw.get("text", "")
        rotation = raw.get("rotation", 0)
        rect_width = raw.get("rectWidth", 0) or raw.get("extentsWidth", 0)
        rect_height = raw.get("rectHeight", 0) or raw.get("extentsHeight", 0)
        px, py = position.get("x", 0), position.get("y", 0)
        if rotation == 0:
            bbox = {"min": {"x": px, "y": py - rect_height}, "max": {"x": px + rect_width, "y": py}}
        else:
            corners = [(0, 0), (rect_width, 0), (rect_width, -rect_height), (0, -rect_height)]
            cos_r, sin_r = math.cos(rotation), math.sin(rotation)
            xs = [px + dx * cos_r - dy * sin_r for dx, dy in corners]
            ys = [py + dx * sin_r + dy * cos_r for dx, dy in corners]
            bbox = {"min": {"x": min(xs), "y": min(ys)}, "max": {"x": max(xs), "y": max(ys)}}
        return {**common, "position": position, "text_content": text_content, "height": height,
                "rotation": rotation, "rect_width": rect_width, "rect_height": rect_height,
                "extents_width": raw.get("extentsWidth", 0), "extents_height": raw.get("extentsHeight", 0),
                "attachment_point": raw.get("attachmentPoint", 1), "drawing_direction": raw.get("drawingDirection", 1),
                "style_name": raw.get("styleName", ""), "line_spacing": raw.get("lineSpacing", 1), "bounding_box": bbox}


class LineExtractor(BaseEntityExtractor):
    ENTITY_TYPE = "LINE"
    def extract(self, entity_info: Any) -> dict:
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        start_point, end_point = point_to_dict(raw.get("startPoint")), point_to_dict(raw.get("endPoint"))
        return {**common, "start_point": start_point, "end_point": end_point,
                "angle": calculate_line_angle(start_point, end_point),
                "length": calculate_line_length(start_point, end_point),
                "thickness": raw.get("thickness", 0),
                "bounding_box": calculate_bounding_box_line(start_point, end_point)}


class CircleExtractor(BaseEntityExtractor):
    ENTITY_TYPE = "CIRCLE"
    def extract(self, entity_info: Any) -> dict:
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        center, radius = point_to_dict(raw.get("center")), raw.get("radius", 0)
        return {**common, "center": center, "radius": radius, "thickness": raw.get("thickness", 0),
                "bounding_box": calculate_bounding_box_circle(center, radius)}


class ArcExtractor(BaseEntityExtractor):
    ENTITY_TYPE = "ARC"
    def extract(self, entity_info: Any) -> dict:
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        center, radius = point_to_dict(raw.get("center")), raw.get("radius", 0)
        start_angle, end_angle = raw.get("startAngle", 0), raw.get("endAngle", 0)
        start_point, end_point = calculate_arc_endpoints(center, radius, start_angle, end_angle)
        return {**common, "start_point": start_point, "end_point": end_point, "center": center,
                "radius": radius, "start_angle": start_angle, "end_angle": end_angle,
                "total_angle": calculate_arc_total_angle(start_angle, end_angle),
                "arc_length": calculate_arc_length(radius, start_angle, end_angle),
                "thickness": raw.get("thickness", 0),
                "bounding_box": calculate_bounding_box_arc(center, radius, start_angle, end_angle)}


class LwPolylineExtractor(BaseEntityExtractor):
    ENTITY_TYPE = "LWPOLYLINE"
    def extract(self, entity_info: Any) -> dict:
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        vertices = []
        for v in raw.get("vertices", []):
            vertex = {"x": v.get("x", 0), "y": v.get("y", 0), "bulge": v.get("bulge", 0), "id": v.get("id", 0)}
            if "startWidth" in v: vertex["start_width"] = v["startWidth"]
            if "endWidth" in v: vertex["end_width"] = v["endWidth"]
            vertices.append(vertex)
        constant_width = raw.get("constantWidth", 0)
        start_width = vertices[0].get("start_width", constant_width) if vertices else constant_width
        end_width = vertices[-1].get("end_width", constant_width) if vertices else constant_width
        flag = raw.get("flag", 0)
        return {**common, "vertices": vertices, "start_width": start_width, "end_width": end_width,
                "constant_width": constant_width, "is_closed": is_polyline_closed(flag),
                "vertex_count": len(vertices), "flag": flag, "elevation": raw.get("elevation", 0),
                "thickness": raw.get("thickness", 0),
                "bounding_box": calculate_bounding_box_points([{"x": v["x"], "y": v["y"]} for v in vertices])}


EXTRACTOR_REGISTRY = {"TEXT": TextExtractor, "MTEXT": MTextExtractor, "LINE": LineExtractor,
                      "CIRCLE": CircleExtractor, "ARC": ArcExtractor, "LWPOLYLINE": LwPolylineExtractor}
_extractor_instances = {}

def get_extractor(entity_type: str):
    entity_type = entity_type.upper()
    if entity_type not in _extractor_instances:
        cls = EXTRACTOR_REGISTRY.get(entity_type)
        if cls: _extractor_instances[entity_type] = cls()
    return _extractor_instances.get(entity_type)

def register_extractor(entity_type: str, extractor_cls):
    entity_type = entity_type.upper()
    EXTRACTOR_REGISTRY[entity_type] = extractor_cls
    _extractor_instances.pop(entity_type, None)


# ============================================================
# 主类 DwgParser
# ============================================================

class DwgParser:
    """
    CAD图纸解析器 - 主入口类
    
    支持多种初始化方式：
    1. 从文件路径: DwgParser(json_path="path/to/file.json") 或 DwgParser("path/to/file.json")
    2. 从字典: DwgParser(json_data={"tables": {...}})
    3. 从字节: DwgParser(json_bytes=b'{"tables": {...}}')
    """
    
    def __init__(
        self, 
        json_path: Optional[str] = None,
        json_data: Optional[dict] = None,
        json_bytes: Optional[bytes] = None
    ):
        self.loader = DwgJsonLoader(json_path=json_path, json_data=json_data, json_bytes=json_bytes)
        if json_path and self.loader._raw_data is None:
            self.loader.load()
        self.entity_manager = EntityManager(self.loader)
        self.layer_manager = LayerManager(self.loader)
    
    @classmethod
    def from_file(cls, json_path: str) -> 'DwgParser':
        return cls(json_path=json_path)
    
    @classmethod
    def from_dict(cls, json_data: dict) -> 'DwgParser':
        return cls(json_data=json_data)
    
    @classmethod
    def from_bytes(cls, json_bytes: bytes) -> 'DwgParser':
        return cls(json_bytes=json_bytes)
    
    @classmethod
    def from_response(cls, response) -> 'DwgParser':
        return cls(json_bytes=response.content)
    
    def get_all_layer_names(self) -> list:
        return self.layer_manager.get_all_layer_names()
    
    def match_layers(self, pattern: str) -> list:
        return self.layer_manager.match_layers(pattern)
    
    def get_all_entities(self, include_nested: bool = True) -> list:
        return self.entity_manager.get_all_entities(include_nested)
    
    def get_entities_by_type(self, entity_type: str) -> list:
        return self.entity_manager.get_entities_by_type(entity_type)
    
    def get_entities_by_layers(self, layer_names: list) -> list:
        return self.entity_manager.get_entities_by_layer(layer_names)
    
    def extract_entity_attrs(self, entity_info: EntityInfo) -> Optional[dict]:
        extractor = get_extractor(entity_info.entity_type)
        return extractor.extract(entity_info) if extractor else None
    
    def get_entity_attr(self, entity_info: EntityInfo, attr_name: str) -> Any:
        attrs = self.extract_entity_attrs(entity_info)
        return attrs.get(attr_name) if attrs else None
    
    def get_supported_entity_types(self) -> list:
        return list(EXTRACTOR_REGISTRY.keys())


__all__ = ['DwgParser', 'DwgJsonLoader', 'EntityManager', 'EntityInfo', 'LayerManager',
           'get_extractor', 'register_extractor', 'EXTRACTOR_REGISTRY', 'BaseEntityExtractor']