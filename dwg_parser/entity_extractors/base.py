"""
实体属性提取器基类

定义所有实体提取器的公共接口和通用属性提取逻辑
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseEntityExtractor(ABC):
    """
    实体属性提取器基类
    
    所有具体的实体提取器都应继承此类并实现extract方法
    
    子类需要：
    1. 设置 ENTITY_TYPE 类属性
    2. 实现 extract 方法
    """
    
    # 子类必须定义此属性，表示处理的实体类型
    ENTITY_TYPE: str = ""
    
    @abstractmethod
    def extract(self, entity_info: Any) -> dict:
        """
        提取实体属性
        
        Args:
            entity_info: 实体信息对象 (EntityInfo)
            
        Returns:
            属性字典，包含该类型实体的所有相关属性
        """
        pass
    
    def _extract_common_attrs(self, entity_info: Any) -> dict:
        """
        提取所有实体共有的属性
        
        Args:
            entity_info: 实体信息对象 (EntityInfo)
            
        Returns:
            通用属性字典，包含：
            - layer: 所在图层
            - block_path: 完整块嵌套路径
            - owner_block: 直接所属块
            - color: 颜色索引
            - color_name: 颜色名称
            - handle: 实体句柄
            - is_visible: 是否可见
            - line_type: 线型
            - line_weight: 线宽
        """
        raw = entity_info.raw_data
        
        return {
            "layer": entity_info.layer,
            "block_path": entity_info.block_path.copy(),
            "owner_block": entity_info.owner_block,
            "color": raw.get("colorIndex", 256),
            "color_name": raw.get("colorName", ""),
            "handle": entity_info.handle,
            "is_visible": raw.get("isVisible", True),
            "line_type": raw.get("lineType", ""),
            "line_weight": raw.get("lineweight", 0),
        }
    
    def _get_bounding_box(self, entity_info: Any) -> dict:
        """
        获取实体的包围盒
        
        子类应重写此方法以提供更精确的包围盒计算
        
        Args:
            entity_info: 实体信息对象 (EntityInfo)
            
        Returns:
            包围盒字典 {"min": {"x", "y"}, "max": {"x", "y"}}
        """
        return {"min": {"x": 0, "y": 0}, "max": {"x": 0, "y": 0}}
