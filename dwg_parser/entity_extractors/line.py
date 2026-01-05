"""
LINE实体属性提取器
"""

from typing import Any
from dwg_parser.entity_extractors.base import BaseEntityExtractor
from dwg_parser.utils import (
    calculate_line_angle, 
    calculate_line_length, 
    calculate_bounding_box_line,
    point_to_dict
)


class LineExtractor(BaseEntityExtractor):
    """
    LINE实体属性提取器
    
    提取属性：
    - start_point: 起点坐标
    - end_point: 终点坐标
    - angle: 直线角度（弧度）
    - length: 直线长度
    - 以及所有通用属性
    """
    
    ENTITY_TYPE = "LINE"
    
    def extract(self, entity_info: Any) -> dict:
        """
        提取LINE实体属性
        
        Args:
            entity_info: 实体信息对象
            
        Returns:
            属性字典
        """
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        
        start_point = point_to_dict(raw.get("startPoint"))
        end_point = point_to_dict(raw.get("endPoint"))
        
        # 计算角度和长度
        angle = calculate_line_angle(start_point, end_point)
        length = calculate_line_length(start_point, end_point)
        
        # 计算包围盒
        bounding_box = calculate_bounding_box_line(start_point, end_point)
        
        return {
            **common,
            "start_point": start_point,
            "end_point": end_point,
            "angle": angle,
            "length": length,
            "thickness": raw.get("thickness", 0),
            "bounding_box": bounding_box,
        }
    
    def _get_bounding_box(self, entity_info: Any) -> dict:
        """计算LINE的包围盒"""
        raw = entity_info.raw_data
        start_point = point_to_dict(raw.get("startPoint"))
        end_point = point_to_dict(raw.get("endPoint"))
        return calculate_bounding_box_line(start_point, end_point)
