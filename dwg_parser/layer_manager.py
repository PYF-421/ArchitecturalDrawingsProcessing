"""
图层管理模块

负责管理和查询图层信息
"""

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dwg_parser.loader import DwgJsonLoader


class LayerManager:
    """
    图层管理器
    
    提供图层查询和正则匹配功能
    """
    
    def __init__(self, loader: 'DwgJsonLoader'):
        """
        初始化图层管理器
        
        Args:
            loader: DwgJsonLoader实例
        """
        self.loader = loader
        self._layer_names_cache: Optional[list[str]] = None
        self._layer_map_cache: Optional[dict[str, dict]] = None
    
    def get_all_layer_names(self) -> list[str]:
        """
        获取全部图层名
        
        Returns:
            图层名列表（过滤掉空名称）
        """
        if self._layer_names_cache is None:
            layers = self.loader.get_layers()
            self._layer_names_cache = [
                layer.get("name", "") 
                for layer in layers 
                if layer.get("name", "")
            ]
        return self._layer_names_cache
    
    def get_layer_map(self) -> dict[str, dict]:
        """
        获取图层名到图层信息的映射
        
        Returns:
            {图层名: 图层信息字典} 的字典
        """
        if self._layer_map_cache is None:
            layers = self.loader.get_layers()
            self._layer_map_cache = {
                layer.get("name", ""): layer
                for layer in layers
                if layer.get("name", "")
            }
        return self._layer_map_cache
    
    def match_layers(self, pattern: str) -> list[str]:
        """
        使用正则表达式匹配图层名
        
        Args:
            pattern: 正则表达式字符串
            
        Returns:
            匹配上的图层名列表
            
        Raises:
            re.error: 正则表达式语法错误
        """
        all_layers = self.get_all_layer_names()
        compiled_pattern = re.compile(pattern)
        return [name for name in all_layers if compiled_pattern.search(name)]
    
    def match_layers_fullmatch(self, pattern: str) -> list[str]:
        """
        使用正则表达式完全匹配图层名
        
        Args:
            pattern: 正则表达式字符串
            
        Returns:
            完全匹配的图层名列表
        """
        all_layers = self.get_all_layer_names()
        compiled_pattern = re.compile(pattern)
        return [name for name in all_layers if compiled_pattern.fullmatch(name)]
    
    def get_layer_info(self, layer_name: str) -> Optional[dict]:
        """
        获取指定图层的详细信息
        
        Args:
            layer_name: 图层名
            
        Returns:
            图层信息字典，不存在则返回None
        """
        return self.get_layer_map().get(layer_name)
    
    def layer_exists(self, layer_name: str) -> bool:
        """
        检查图层是否存在
        
        Args:
            layer_name: 图层名
            
        Returns:
            是否存在
        """
        return layer_name in self.get_layer_map()
    
    def get_layer_color(self, layer_name: str) -> Optional[int]:
        """
        获取图层颜色索引
        
        Args:
            layer_name: 图层名
            
        Returns:
            颜色索引值，图层不存在则返回None
        """
        layer_info = self.get_layer_info(layer_name)
        if layer_info:
            return layer_info.get("colorIndex")
        return None
    
    def is_layer_frozen(self, layer_name: str) -> Optional[bool]:
        """
        检查图层是否冻结
        
        Args:
            layer_name: 图层名
            
        Returns:
            是否冻结，图层不存在则返回None
        """
        layer_info = self.get_layer_info(layer_name)
        if layer_info:
            return layer_info.get("frozen", False)
        return None
    
    def is_layer_off(self, layer_name: str) -> Optional[bool]:
        """
        检查图层是否关闭
        
        Args:
            layer_name: 图层名
            
        Returns:
            是否关闭，图层不存在则返回None
        """
        layer_info = self.get_layer_info(layer_name)
        if layer_info:
            return layer_info.get("off", False)
        return None
    
    def is_layer_locked(self, layer_name: str) -> Optional[bool]:
        """
        检查图层是否锁定
        
        Args:
            layer_name: 图层名
            
        Returns:
            是否锁定，图层不存在则返回None
        """
        layer_info = self.get_layer_info(layer_name)
        if layer_info:
            return layer_info.get("locked", False)
        return None
    
    def get_visible_layers(self) -> list[str]:
        """
        获取所有可见图层（未冻结且未关闭）
        
        Returns:
            可见图层名列表
        """
        visible = []
        for name, info in self.get_layer_map().items():
            if not info.get("frozen", False) and not info.get("off", False):
                visible.append(name)
        return visible
