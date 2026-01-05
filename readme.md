# ArchitecturalDrawingsProcessing 

主要功能：

- 从DWG导出的JSON文件中解析图层、实体（LINE、TEXT、MTEXT、LWPOLYLINE、CIRCLE、ARC、INSERT等）。
- 提供便捷的图层管理和实体提取接口。
- 提供RabbitMQ消费者示例（`consumer.py`）用于生产环境集成。
- 包含一个简单的Flask健康检查服务（`app.py`）。

核心模块：
- `dwg_parser`：底层解析器，提供加载、实体管理和图层管理。

---

## 模块：dwg_parser

### 类：DwgJsonLoader

**作用**：负责加载DWG JSON数据，支持多种来源。

#### 构造方法
```python
DwgJsonLoader(
    json_path: Optional[str] = None,
    json_data: Optional[dict] = None,
    json_bytes: Optional[bytes] = None
)
```

#### 类方法
- `from_file(json_path: str) -> DwgJsonLoader`
- `from_dict(json_data: dict) -> DwgJsonLoader`
- `from_bytes(json_bytes: bytes) -> DwgJsonLoader`
- `from_response(response) -> DwgJsonLoader`  # response为requests响应对象

#### 实例方法
- `load() -> dict`：从文件路径加载并返回原始JSON字典。
- `get_raw_data() -> dict`：获取已加载的原始数据。
- `get_tables() -> dict`：获取tables部分。
- `get_entities() -> list`：获取模型空间顶层entities。
- `get_layers() -> list`：获取LAYER表条目。
- `get_block_records() -> list`：获取BLOCK_RECORD表条目。
- `get_block_by_name(name: str) -> Optional[dict]`
- `get_block_by_handle(handle: int) -> Optional[dict]`

### 类：LayerManager

**作用**：管理图层信息。

#### 构造方法
```python
LayerManager(loader: DwgJsonLoader)
```

#### 方法
- `get_all_layer_names() -> list[str]`：返回所有图层名。
- `match_layers(pattern: str) -> list[str]`：正则搜索图层名（`re.search`）。
- `match_layers_fullmatch(pattern: str) -> list[str]`：正则完全匹配图层名。
- `layer_exists(layer_name: str) -> bool`
- `get_layer_info(layer_name: str) -> Optional[dict]`

### 类：EntityManager

**作用**：管理所有实体（包括块内嵌套实体），提供统一访问接口。

#### 构造方法
```python
EntityManager(loader: DwgJsonLoader)
```

#### 方法
- `get_all_entities(include_nested: bool = True) -> list[EntityInfo]`：
  - 返回所有实体（`EntityInfo`对象）。
  - `include_nested=False` 时只返回模型空间实体。
- `get_entities_by_type(entity_type: str) -> list[EntityInfo]`
- `get_entities_by_layer(layer_names: list[str]) -> list[EntityInfo]`
- `get_entities_by_type_and_layer(entity_type: str, layer_names: list[str]) -> list[EntityInfo]`
- `count_entities_by_type() -> dict[str, int]`：按类型统计数量。

#### dataclasses：EntityInfo
```python
@dataclass
class EntityInfo:
    raw_data: dict
    entity_type: str
    handle: int
    layer: str
    block_path: list[str] = field(default_factory=list)
    owner_block: str = ""
```

### 主类：DwgParser（通过__init__.py导出）

**作用**：高层封装，提供最常用的解析接口。

#### 构造方法（示例）
```python
DwgParser(json_path: str)  # 或通过类方法
```

#### 常用方法（基于代码引用）
- `get_all_layer_names() -> list[str]`
- `match_layers(pattern: str) -> list[str]`
- `get_all_entities(include_nested: bool = True) -> list[EntityInfo]`
- `get_entities_by_type(entity_type: str) -> list[EntityInfo]`
- `get_entities_by_layers(layer_names: list[str]) -> list[EntityInfo]`
- `extract_entity_attrs(entity_info: EntityInfo) -> dict`：提取标准化属性（位置、文字内容、角度、长度等）。

### 类：LayerManager（独立实现）

#### 方法
- `FindLayerName(pattern: str) -> list[str]`：正则匹配图层（`re.compile(pattern).match`）。
- `list_all_layers() -> list[str]`：排序后的所有图层名。

### 类：EntityExtractor

#### 构造方法
```python
EntityExtractor(json_path: str = None, data: dict = None, layer_name: str = None)
```

#### 特殊调用
- `extractor(layer_name: str)` → 返回限定该图层的提取器实例。

#### 实体提取方法（返回EntityCollection）
- `MTEXT() -> EntityCollection`
- `TEXT() -> EntityCollection`
- `LINE() -> EntityCollection`
- `LWPOLYLINE() -> EntityCollection`
- `CIRCLE() -> EntityCollection`
- `ARC() -> EntityCollection`
- `INSERT() -> EntityCollection`

### 类：EntityCollection

#### 方法
- `gettexts() -> list[str]`：获取所有文字内容（TEXT/MTEXT通用）。
- `gettext() -> str`：获取第一个文字内容。
- `getpositions() -> list[dict]`：获取插入点位置。
- `getlayers() -> list[str]`：获取实体图层。
- `raw() -> list[dict]`：返回原始实体数据。
- `__len__()`、`__getitem__()`：支持长度和索引访问。

#### 使用示例
```python
extractor = EntityExtractor("test.json")
texts = extractor.MTEXT().gettexts()
beam_texts = extractor("BEAM-JZ-V").TEXT().gettexts()
```

### 配置（修改CONFIG类）
- `JSON_PATH`：输入JSON路径
- `OUTPUT_JSON`：输出修改后JSON路径

---

## 使用建议

1. **基础解析**：使用`DwgParser`或`EntityExtractor`快速提取实体。
4. **生产集成**：参考`consumer.py`使用RabbitMQ消费任务。

如需进一步扩展或有具体问题，可根据接口添加新实体提取器或匹配规则。