"""
LWPOLYLINE实体属性提取器
"""

from typing import Any
from dwg_parser.entity_extractors.base import BaseEntityExtractor
from dwg_parser.utils import (
    is_polyline_closed,
    calculate_bounding_box_points,
    point_to_dict
)


class LwPolylineExtractor(BaseEntityExtractor):
    """
    LWPOLYLINE实体属性提取器
    
    LWPOLYLINE是轻量级多段线
    
    提取属性：
    - vertices: 顶点坐标列表
    - start_width: 起始线段宽度（从第一个顶点获取）
    - end_width: 终止线段宽度（从最后一个顶点获取）
    - constant_width: 全局宽度
    - is_closed: 是否闭合
    - vertex_count: 顶点数量
    - 以及所有通用属性
    """
    
    ENTITY_TYPE = "LWPOLYLINE"
    
    def extract(self, entity_info: Any) -> dict:
        """
        提取LWPOLYLINE实体属性
        
        Args:
            entity_info: 实体信息对象
            
        Returns:
            属性字典
        """
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        
        # 提取顶点
        raw_vertices = raw.get("vertices", [])
        vertices = []
        for v in raw_vertices:
            vertex = {
                "x": v.get("x", 0),
                "y": v.get("y", 0),
                "bulge": v.get("bulge", 0),
                "id": v.get("id", 0)
            }
            # 如果顶点有单独的宽度信息
            if "startWidth" in v:
                vertex["start_width"] = v.get("startWidth", 0)
            if "endWidth" in v:
                vertex["end_width"] = v.get("endWidth", 0)
            vertices.append(vertex)
        
        # 获取宽度信息
        constant_width = raw.get("constantWidth", 0)
        
        # 起始和终止宽度（从顶点获取，如果没有则使用常量宽度）
        start_width = constant_width
        end_width = constant_width
        if vertices:
            if "start_width" in vertices[0]:
                start_width = vertices[0]["start_width"]
            if "end_width" in vertices[-1]:
                end_width = vertices[-1]["end_width"]
        
        # 判断是否闭合
        flag = raw.get("flag", 0)
        is_closed = is_polyline_closed(flag)
        
        # 计算包围盒
        points = [{"x": v["x"], "y": v["y"]} for v in vertices]
        bounding_box = calculate_bounding_box_points(points)
        
        return {
            **common,
            "vertices": vertices,
            "start_width": start_width,
            "end_width": end_width,
            "constant_width": constant_width,
            "is_closed": is_closed,
            "vertex_count": len(vertices),
            "flag": flag,
            "elevation": raw.get("elevation", 0),
            "thickness": raw.get("thickness", 0),
            "bounding_box": bounding_box,
        }
    
    def _get_bounding_box(self, entity_info: Any) -> dict:
        """计算LWPOLYLINE的包围盒"""
        raw = entity_info.raw_data
        raw_vertices = raw.get("vertices", [])
        points = [{"x": v.get("x", 0), "y": v.get("y", 0)} for v in raw_vertices]
        return calculate_bounding_box_points(points)
