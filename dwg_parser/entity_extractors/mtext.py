"""
MTEXT实体属性提取器
"""

from typing import Any
from dwg_parser.entity_extractors.base import BaseEntityExtractor
from dwg_parser.utils import calculate_bounding_box_text, point_to_dict


class MTextExtractor(BaseEntityExtractor):
    """
    MTEXT实体属性提取器
    
    MTEXT是多行文字，具有更多的格式控制能力
    
    提取属性：
    - position: 坐标信息（insertionPoint）
    - text_content: 文字内容
    - height: 文字高度
    - rotation: 旋转角度（弧度）
    - rect_width: 文字框宽度
    - rect_height: 文字框高度
    - attachment_point: 对齐点（1-9）
    - 以及所有通用属性
    """
    
    ENTITY_TYPE = "MTEXT"
    
    def extract(self, entity_info: Any) -> dict:
        """
        提取MTEXT实体属性
        
        Args:
            entity_info: 实体信息对象
            
        Returns:
            属性字典
        """
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        
        position = point_to_dict(raw.get("insertionPoint"))
        height = raw.get("textHeight", 0)
        text_content = raw.get("text", "")
        rotation = raw.get("rotation", 0)
        rect_width = raw.get("rectWidth", 0)
        rect_height = raw.get("rectHeight", 0)
        
        # 使用extentsWidth和extentsHeight如果rectWidth/rectHeight为0
        if rect_width == 0:
            rect_width = raw.get("extentsWidth", 0)
        if rect_height == 0:
            rect_height = raw.get("extentsHeight", 0)
        
        # 计算包围盒
        bounding_box = self._calculate_mtext_bounding_box(
            position, rect_width, rect_height, rotation
        )
        
        return {
            **common,
            "position": position,
            "text_content": text_content,
            "height": height,
            "rotation": rotation,
            "rect_width": rect_width,
            "rect_height": rect_height,
            "extents_width": raw.get("extentsWidth", 0),
            "extents_height": raw.get("extentsHeight", 0),
            "attachment_point": raw.get("attachmentPoint", 1),
            "drawing_direction": raw.get("drawingDirection", 1),
            "style_name": raw.get("styleName", ""),
            "line_spacing": raw.get("lineSpacing", 1),
            "bounding_box": bounding_box,
        }
    
    def _calculate_mtext_bounding_box(
        self, 
        position: dict, 
        width: float, 
        height: float, 
        rotation: float
    ) -> dict:
        """
        计算MTEXT的包围盒
        
        Args:
            position: 插入点
            width: 宽度
            height: 高度
            rotation: 旋转角度
            
        Returns:
            包围盒字典
        """
        import math
        
        px = position.get("x", 0)
        py = position.get("y", 0)
        
        if rotation == 0:
            return {
                "min": {"x": px, "y": py - height},
                "max": {"x": px + width, "y": py}
            }
        
        # 考虑旋转
        corners = [
            (0, 0),
            (width, 0),
            (width, -height),
            (0, -height)
        ]
        
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        
        xs = []
        ys = []
        for dx, dy in corners:
            rx = px + dx * cos_r - dy * sin_r
            ry = py + dx * sin_r + dy * cos_r
            xs.append(rx)
            ys.append(ry)
        
        return {
            "min": {"x": min(xs), "y": min(ys)},
            "max": {"x": max(xs), "y": max(ys)}
        }
    
    def _get_bounding_box(self, entity_info: Any) -> dict:
        """计算MTEXT的包围盒"""
        raw = entity_info.raw_data
        position = point_to_dict(raw.get("insertionPoint"))
        rect_width = raw.get("rectWidth", 0) or raw.get("extentsWidth", 0)
        rect_height = raw.get("rectHeight", 0) or raw.get("extentsHeight", 0)
        rotation = raw.get("rotation", 0)
        
        return self._calculate_mtext_bounding_box(
            position, rect_width, rect_height, rotation
        )
