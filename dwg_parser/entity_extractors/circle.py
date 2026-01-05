"""
CIRCLE实体属性提取器
"""

from typing import Any
from dwg_parser.entity_extractors.base import BaseEntityExtractor
from dwg_parser.utils import calculate_bounding_box_circle, point_to_dict


class CircleExtractor(BaseEntityExtractor):
    """
    CIRCLE实体属性提取器
    
    提取属性：
    - center: 圆心坐标
    - radius: 半径
    - 以及所有通用属性
    """
    
    ENTITY_TYPE = "CIRCLE"
    
    def extract(self, entity_info: Any) -> dict:
        """
        提取CIRCLE实体属性
        
        Args:
            entity_info: 实体信息对象
            
        Returns:
            属性字典
        """
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        
        center = point_to_dict(raw.get("center"))
        radius = raw.get("radius", 0)
        
        # 计算包围盒
        bounding_box = calculate_bounding_box_circle(center, radius)
        
        return {
            **common,
            "center": center,
            "radius": radius,
            "thickness": raw.get("thickness", 0),
            "bounding_box": bounding_box,
        }
    
    def _get_bounding_box(self, entity_info: Any) -> dict:
        """计算CIRCLE的包围盒"""
        raw = entity_info.raw_data
        center = point_to_dict(raw.get("center"))
        radius = raw.get("radius", 0)
        return calculate_bounding_box_circle(center, radius)
