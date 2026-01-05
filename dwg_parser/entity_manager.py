"""
实体管理模块

负责提取和管理所有实体，包括处理块嵌套关系
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dwg_parser.loader import DwgJsonLoader


@dataclass
class EntityInfo:
    """
    实体信息包装类
    
    包含实体的原始数据和元信息（块嵌套关系等）
    
    Attributes:
        raw_data: 原始JSON数据
        entity_type: 实体类型（如 "TEXT", "LINE" 等）
        handle: 实体句柄
        layer: 所在图层名
        block_path: 块嵌套路径，从根块到直接所属块的完整路径
                   例如: ["*Model_Space", "k5", "_ArchTick"]
        owner_block: 直接所属块名
    """
    raw_data: dict
    entity_type: str
    handle: int
    layer: str
    block_path: list[str] = field(default_factory=list)
    owner_block: str = ""
    
    def __post_init__(self):
        """初始化后处理"""
        if self.block_path:
            self.owner_block = self.block_path[-1]
    
    def is_in_model_space(self) -> bool:
        """判断实体是否直接在Model Space中"""
        return len(self.block_path) == 1 and self.block_path[0] == "*Model_Space"
    
    def get_nesting_depth(self) -> int:
        """获取块嵌套深度（0表示直接在Model Space中）"""
        return len(self.block_path) - 1


class EntityManager:
    """
    实体管理器
    
    负责提取和管理所有实体，处理块嵌套关系
    """
    
    def __init__(self, loader: 'DwgJsonLoader'):
        """
        初始化实体管理器
        
        Args:
            loader: DwgJsonLoader实例
        """
        self.loader = loader
        self._all_entities_cache: Optional[list[EntityInfo]] = None
    
    def _extract_entities_from_block(
        self, 
        block: dict, 
        current_path: list[str]
    ) -> list[EntityInfo]:
        """
        从块中递归提取所有实体
        
        Args:
            block: 块记录字典
            current_path: 当前块路径
            
        Returns:
            实体信息列表
        """
        entities = []
        block_name = block.get("name", "")
        new_path = current_path + [block_name]
        
        for entity_data in block.get("entities", []):
            entity_type = entity_data.get("type", "")
            
            # 创建EntityInfo
            entity_info = EntityInfo(
                raw_data=entity_data,
                entity_type=entity_type,
                handle=entity_data.get("handle", 0),
                layer=entity_data.get("layer", ""),
                block_path=new_path.copy()
            )
            entities.append(entity_info)
            
            # 如果是INSERT（块引用），递归处理引用的块
            if entity_type == "INSERT":
                insert_block_name = entity_data.get("name", "")
                if insert_block_name:
                    referenced_block = self.loader.get_block_by_name(insert_block_name)
                    if referenced_block:
                        nested_entities = self._extract_entities_from_block(
                            referenced_block, 
                            new_path
                        )
                        entities.extend(nested_entities)
        
        return entities
    
    def get_all_entities(self, include_nested: bool = True) -> list[EntityInfo]:
        """
        获取所有实体
        
        Args:
            include_nested: 是否包含嵌套在块中的实体
            
        Returns:
            实体信息列表
        """
        if self._all_entities_cache is not None:
            if include_nested:
                return self._all_entities_cache
            else:
                return [e for e in self._all_entities_cache if e.is_in_model_space()]
        
        entities = []
        
        # 从*Model_Space开始提取
        model_space = self.loader.get_block_by_name("*Model_Space")
        if model_space:
            entities = self._extract_entities_from_block(model_space, [])
        
        self._all_entities_cache = entities
        
        if include_nested:
            return entities
        else:
            return [e for e in entities if e.is_in_model_space()]
    
    def get_entities_by_type(self, entity_type: str) -> list[EntityInfo]:
        """
        按类型获取实体
        
        Args:
            entity_type: 实体类型名（如 "TEXT", "LINE" 等）
            
        Returns:
            符合类型的实体信息列表
        """
        all_entities = self.get_all_entities()
        return [e for e in all_entities if e.entity_type == entity_type.upper()]
    
    def get_entities_by_layer(self, layer_names: list[str]) -> list[EntityInfo]:
        """
        按图层名列表筛选实体
        
        Args:
            layer_names: 图层名列表
            
        Returns:
            在指定图层中的实体信息列表
        """
        all_entities = self.get_all_entities()
        layer_set = set(layer_names)
        return [e for e in all_entities if e.layer in layer_set]
    
    def get_entities_by_type_and_layer(
        self, 
        entity_type: str, 
        layer_names: list[str]
    ) -> list[EntityInfo]:
        """
        同时按类型和图层筛选实体
        
        Args:
            entity_type: 实体类型名
            layer_names: 图层名列表
            
        Returns:
            符合条件的实体信息列表
        """
        all_entities = self.get_all_entities()
        layer_set = set(layer_names)
        return [
            e for e in all_entities 
            if e.entity_type == entity_type.upper() and e.layer in layer_set
        ]
    
    def get_entity_types(self) -> list[str]:
        """
        获取文件中存在的所有实体类型
        
        Returns:
            实体类型名列表
        """
        all_entities = self.get_all_entities()
        return list(set(e.entity_type for e in all_entities))
    
    def count_entities_by_type(self) -> dict[str, int]:
        """
        统计各类型实体的数量
        
        Returns:
            {实体类型: 数量} 的字典
        """
        all_entities = self.get_all_entities()
        counts = {}
        for e in all_entities:
            counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
        return counts
