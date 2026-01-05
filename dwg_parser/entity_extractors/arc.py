"""
ARC实体属性提取器
"""

from typing import Any
from dwg_parser.entity_extractors.base import BaseEntityExtractor
from dwg_parser.utils import (
    calculate_arc_endpoints,
    calculate_arc_total_angle,
    calculate_arc_length,
    calculate_bounding_box_arc,
    point_to_dict
)


class ArcExtractor(BaseEntityExtractor):
    """
    ARC实体属性提取器
    
    提取属性：
    - start_point: 起点坐标（计算得出）
    - end_point: 终点坐标（计算得出）
    - center: 圆心坐标
    - radius: 半径
    - start_angle: 起始角度（弧度）
    - end_angle: 终止角度（弧度）
    - total_angle: 总角度（弧度）
    - arc_length: 弧长
    - 以及所有通用属性
    """
    
    ENTITY_TYPE = "ARC"
    
    def extract(self, entity_info: Any) -> dict:
        """
        提取ARC实体属性
        
        Args:
            entity_info: 实体信息对象
            
        Returns:
            属性字典
        """
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        
        center = point_to_dict(raw.get("center"))
        radius = raw.get("radius", 0)
        start_angle = raw.get("startAngle", 0)
        end_angle = raw.get("endAngle", 0)
        
        # 计算起止点
        start_point, end_point = calculate_arc_endpoints(
            center, radius, start_angle, end_angle
        )
        
        # 计算总角度和弧长
        total_angle = calculate_arc_total_angle(start_angle, end_angle)
        arc_length = calculate_arc_length(radius, start_angle, end_angle)
        
        # 计算包围盒
        bounding_box = calculate_bounding_box_arc(
            center, radius, start_angle, end_angle
        )
        
        return {
            **common,
            "start_point": start_point,
            "end_point": end_point,
            "center": center,
            "radius": radius,
            "start_angle": start_angle,
            "end_angle": end_angle,
            "total_angle": total_angle,
            "arc_length": arc_length,
            "thickness": raw.get("thickness", 0),
            "bounding_box": bounding_box,
        }
    
    def _get_bounding_box(self, entity_info: Any) -> dict:
        """计算ARC的包围盒"""
        raw = entity_info.raw_data
        center = point_to_dict(raw.get("center"))
        radius = raw.get("radius", 0)
        start_angle = raw.get("startAngle", 0)
        end_angle = raw.get("endAngle", 0)
        return calculate_bounding_box_arc(center, radius, start_angle, end_angle)
