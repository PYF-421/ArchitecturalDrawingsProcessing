"""
JSON加载器模块

负责加载和解析libredwg导出的JSON文件
支持从文件路径、字典或字节数据加载
"""

import json
from typing import Optional, Union


class DwgJsonLoader:
    """
    DWG JSON加载器
    
    加载并解析libredwg导出的JSON文件，提供对各部分数据的访问接口
    
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
        """
        初始化加载器
        
        Args:
            json_path: JSON文件路径
            json_data: 已解析的JSON字典数据
            json_bytes: JSON字节数据
            
        注意：三个参数只需提供一个，优先级为 json_data > json_bytes > json_path
        """
        self.json_path = json_path
        self._raw_data: Optional[dict] = None
        self._block_records_cache: Optional[list] = None
        self._block_map_cache: Optional[dict] = None
        
        # 如果直接传入了数据，立即加载
        if json_data is not None:
            self._raw_data = json_data
        elif json_bytes is not None:
            self._raw_data = json.loads(json_bytes.decode('utf-8'))
    
    @classmethod
    def from_file(cls, json_path: str) -> 'DwgJsonLoader':
        """
        从文件路径创建加载器
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            DwgJsonLoader实例
        """
        loader = cls(json_path=json_path)
        loader.load()
        return loader
    
    @classmethod
    def from_dict(cls, json_data: dict) -> 'DwgJsonLoader':
        """
        从字典数据创建加载器
        
        Args:
            json_data: 已解析的JSON字典
            
        Returns:
            DwgJsonLoader实例
        """
        return cls(json_data=json_data)
    
    @classmethod
    def from_bytes(cls, json_bytes: bytes) -> 'DwgJsonLoader':
        """
        从字节数据创建加载器
        
        Args:
            json_bytes: JSON字节数据
            
        Returns:
            DwgJsonLoader实例
        """
        return cls(json_bytes=json_bytes)
    
    @classmethod
    def from_response(cls, response) -> 'DwgJsonLoader':
        """
        从requests响应对象创建加载器
        
        Args:
            response: requests.Response对象
            
        Returns:
            DwgJsonLoader实例
        """
        return cls(json_bytes=response.content)
    
    def load(self) -> dict:
        """
        从文件加载JSON
        
        Returns:
            解析后的JSON数据字典
            
        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
            ValueError: 未指定文件路径
        """
        if self._raw_data is not None:
            return self._raw_data
            
        if self.json_path is None:
            raise ValueError("未指定JSON文件路径，且未提供数据")
            
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self._raw_data = json.load(f)
        return self._raw_data
    
    def load_from_dict(self, json_data: dict) -> dict:
        """
        从字典加载数据
        
        Args:
            json_data: JSON字典数据
            
        Returns:
            JSON数据字典
        """
        self._raw_data = json_data
        self._clear_cache()
        return self._raw_data
    
    def load_from_bytes(self, json_bytes: bytes) -> dict:
        """
        从字节数据加载
        
        Args:
            json_bytes: JSON字节数据
            
        Returns:
            JSON数据字典
        """
        self._raw_data = json.loads(json_bytes.decode('utf-8'))
        self._clear_cache()
        return self._raw_data
    
    def _clear_cache(self):
        """清除所有缓存"""
        self._block_records_cache = None
        self._block_map_cache = None
    
    def get_raw_data(self) -> dict:
        """
        获取原始JSON数据
        
        Returns:
            原始JSON数据字典
        """
        if self._raw_data is None:
            self.load()
        return self._raw_data
    
    def get_tables(self) -> dict:
        """
        获取tables部分数据
        
        Returns:
            tables字典
        """
        return self.get_raw_data().get("tables", {})
    
    def get_block_records(self) -> list:
        """
        获取所有BLOCK_RECORD
        
        Returns:
            BLOCK_RECORD列表
        """
        if self._block_records_cache is None:
            tables = self.get_tables()
            block_record = tables.get("BLOCK_RECORD", {})
            self._block_records_cache = block_record.get("entries", [])
        return self._block_records_cache
    
    def get_block_map(self) -> dict:
        """
        获取块名到块记录的映射
        
        Returns:
            {块名: 块记录} 的字典
        """
        if self._block_map_cache is None:
            self._block_map_cache = {}
            for block in self.get_block_records():
                name = block.get("name", "")
                if name:
                    self._block_map_cache[name] = block
                # 同时用handle作为key，方便通过ownerBlockRecordSoftId查找
                handle = block.get("handle")
                if handle is not None:
                    self._block_map_cache[handle] = block
        return self._block_map_cache
    
    def get_block_by_name(self, name: str) -> Optional[dict]:
        """
        根据块名获取块记录
        
        Args:
            name: 块名
            
        Returns:
            块记录字典，不存在则返回None
        """
        return self.get_block_map().get(name)
    
    def get_block_by_handle(self, handle: int) -> Optional[dict]:
        """
        根据handle获取块记录
        
        Args:
            handle: 块的handle值
            
        Returns:
            块记录字典，不存在则返回None
        """
        return self.get_block_map().get(handle)
    
    def get_layers(self) -> list:
        """
        获取LAYER表中的所有图层
        
        Returns:
            图层列表
        """
        tables = self.get_tables()
        layer_table = tables.get("LAYER", {})
        return layer_table.get("entries", [])
    
    def get_entities(self) -> list:
        """
        获取顶层entities（与*Model_Space中的entities相同）
        
        Returns:
            实体列表
        """
        return self.get_raw_data().get("entities", [])