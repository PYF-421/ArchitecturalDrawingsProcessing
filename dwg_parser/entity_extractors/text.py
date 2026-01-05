"""
TEXT实体属性提取器
"""

from typing import Any
from dwg_parser.entity_extractors.base import BaseEntityExtractor
from dwg_parser.utils import calculate_bounding_box_text, point_to_dict


class TextExtractor(BaseEntityExtractor):
    """
    TEXT实体属性提取器
    
    提取属性：
    - position: 坐标信息（startPoint）
    - text_content: 文字内容
    - height: 文字高度
    - rotation: 旋转角度（弧度）
    - x_scale: X方向缩放比例
    - style_name: 文字样式名
    - 以及所有通用属性
    """
    
    ENTITY_TYPE = "TEXT"
    
    def extract(self, entity_info: Any) -> dict:
        """
        提取TEXT实体属性
        
        Args:
            entity_info: 实体信息对象
            
        Returns:
            属性字典
        """
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        
        position = point_to_dict(raw.get("startPoint"))
        height = raw.get("textHeight", 0)
        text_content = raw.get("text", "")
        rotation = raw.get("rotation", 0)
        x_scale = raw.get("xScale", 1.0)
        
        # 计算包围盒
        bounding_box = calculate_bounding_box_text(
            position, height, text_content, rotation, x_scale
        )
        
        return {
            **common,
            "position": position,
            "text_content": text_content,
            "height": height,
            "rotation": rotation,
            "x_scale": x_scale,
            "style_name": raw.get("styleName", ""),
            "oblique_angle": raw.get("obliqueAngle", 0),
            "h_align": raw.get("halign", 0),
            "v_align": raw.get("valign", 0),
            "bounding_box": bounding_box,
        }
    
    def _get_bounding_box(self, entity_info: Any) -> dict:
        """计算TEXT的包围盒"""
        raw = entity_info.raw_data
        position = point_to_dict(raw.get("startPoint"))
        height = raw.get("textHeight", 0)
        text_content = raw.get("text", "")
        rotation = raw.get("rotation", 0)
        x_scale = raw.get("xScale", 1.0)
        
        return calculate_bounding_box_text(
            position, height, text_content, rotation, x_scale
        )
