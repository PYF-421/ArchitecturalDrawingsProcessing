"""
实体属性提取器模块

提供各种CAD实体的属性提取功能

使用方法：
    from dwg_parser.entity_extractors import get_extractor, register_extractor
    
    # 获取提取器
    extractor = get_extractor("TEXT")
    if extractor:
        attrs = extractor.extract(entity_info)
    
    # 注册自定义提取器
    register_extractor("CUSTOM_ENTITY", CustomExtractor)
"""

from typing import Optional, Type

from dwg_parser.entity_extractors.base import BaseEntityExtractor
from dwg_parser.entity_extractors.text import TextExtractor
from dwg_parser.entity_extractors.mtext import MTextExtractor
from dwg_parser.entity_extractors.line import LineExtractor
from dwg_parser.entity_extractors.circle import CircleExtractor
from dwg_parser.entity_extractors.arc import ArcExtractor
from dwg_parser.entity_extractors.lwpolyline import LwPolylineExtractor


# 提取器注册表：实体类型名 -> 提取器类
EXTRACTOR_REGISTRY: dict[str, Type[BaseEntityExtractor]] = {
    "TEXT": TextExtractor,
    "MTEXT": MTextExtractor,
    "LINE": LineExtractor,
    "CIRCLE": CircleExtractor,
    "ARC": ArcExtractor,
    "LWPOLYLINE": LwPolylineExtractor,
}

# 提取器实例缓存（单例模式）
_extractor_instances: dict[str, BaseEntityExtractor] = {}


def get_extractor(entity_type: str) -> Optional[BaseEntityExtractor]:
    """
    获取对应实体类型的提取器实例
    
    使用单例模式，同一类型的提取器只创建一个实例
    
    Args:
        entity_type: 实体类型名（如 "TEXT", "LINE" 等）
        
    Returns:
        提取器实例，如果该类型没有注册则返回None
    """
    entity_type = entity_type.upper()
    
    # 检查缓存
    if entity_type in _extractor_instances:
        return _extractor_instances[entity_type]
    
    # 获取提取器类
    extractor_cls = EXTRACTOR_REGISTRY.get(entity_type)
    if extractor_cls is None:
        return None
    
    # 创建实例并缓存
    instance = extractor_cls()
    _extractor_instances[entity_type] = instance
    return instance


def register_extractor(entity_type: str, extractor_cls: Type[BaseEntityExtractor]) -> None:
    """
    注册新的实体提取器
    
    用于扩展支持新的实体类型
    
    Args:
        entity_type: 实体类型名
        extractor_cls: 提取器类（必须继承自BaseEntityExtractor）
        
    Raises:
        TypeError: 如果提取器类不是BaseEntityExtractor的子类
        
    Example:
        class HatchExtractor(BaseEntityExtractor):
            ENTITY_TYPE = "HATCH"
            
            def extract(self, entity_info):
                # 实现提取逻辑
                pass
        
        register_extractor("HATCH", HatchExtractor)
    """
    if not issubclass(extractor_cls, BaseEntityExtractor):
        raise TypeError(
            f"extractor_cls must be a subclass of BaseEntityExtractor, "
            f"got {extractor_cls.__name__}"
        )
    
    entity_type = entity_type.upper()
    EXTRACTOR_REGISTRY[entity_type] = extractor_cls
    
    # 清除可能存在的旧实例缓存
    if entity_type in _extractor_instances:
        del _extractor_instances[entity_type]


def unregister_extractor(entity_type: str) -> bool:
    """
    取消注册实体提取器
    
    Args:
        entity_type: 实体类型名
        
    Returns:
        是否成功取消注册（如果类型不存在返回False）
    """
    entity_type = entity_type.upper()
    
    if entity_type in EXTRACTOR_REGISTRY:
        del EXTRACTOR_REGISTRY[entity_type]
        if entity_type in _extractor_instances:
            del _extractor_instances[entity_type]
        return True
    return False


def get_supported_types() -> list[str]:
    """
    获取所有已注册的实体类型
    
    Returns:
        支持的实体类型名列表
    """
    return list(EXTRACTOR_REGISTRY.keys())


def is_type_supported(entity_type: str) -> bool:
    """
    检查实体类型是否已注册
    
    Args:
        entity_type: 实体类型名
        
    Returns:
        是否支持该类型
    """
    return entity_type.upper() in EXTRACTOR_REGISTRY


__all__ = [
    'BaseEntityExtractor',
    'TextExtractor',
    'MTextExtractor',
    'LineExtractor',
    'CircleExtractor',
    'ArcExtractor',
    'LwPolylineExtractor',
    'EXTRACTOR_REGISTRY',
    'get_extractor',
    'register_extractor',
    'unregister_extractor',
    'get_supported_types',
    'is_type_supported',
]