# ArchitecturalDrawingsProcessing

主要功能：

- 从DWG导出的JSON文件中解析图层、实体（LINE、TEXT、MTEXT、LWPOLYLINE、CIRCLE、ARC、INSERT等）。
- 提供便捷的图层管理和实体提取接口。
- 提供块引用(INSERT)管理，支持坐标变换和嵌套块展开。
- 提供RabbitMQ消费者示例（`consumer.py`）用于生产环境集成。
- 包含一个简单的Flask健康检查服务（`app.py`）。

核心模块：

- `dwg_parser`：底层解析器，提供加载、实体管理、图层管理和块引用管理。

------

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

------

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

------

### 类：EntityManager

**作用**：管理所有实体（包括块内嵌套实体），提供统一访问接口。

#### 构造方法

```python
EntityManager(loader: DwgJsonLoader)
```

#### 方法

- ```
  get_all_entities(include_nested: bool = True) -> list[EntityInfo]
  ```

  ：

  - 返回所有实体（`EntityInfo`对象）。
  - `include_nested=False` 时只返回模型空间实体。

- `get_entities_by_type(entity_type: str) -> list[EntityInfo]`

- `get_entities_by_layer(layer_names: list[str]) -> list[EntityInfo]`

- `get_entities_by_type_and_layer(entity_type: str, layer_names: list[str]) -> list[EntityInfo]`

- `count_entities_by_type() -> dict[str, int]`：按类型统计数量。

#### dataclass：EntityInfo

```python
@dataclass
class EntityInfo:
    raw_data: dict          # 原始实体数据
    entity_type: str        # 实体类型 (LINE, TEXT, INSERT等)
    handle: int             # 实体句柄
    layer: str              # 所在图层
    block_path: list[str]   # 块路径 (如 ['*Model_Space', 'BlockA'])
    owner_block: str        # 直接所属块名
    
    def is_in_model_space() -> bool    # 是否在模型空间顶层
    def get_nesting_depth() -> int     # 嵌套深度
```

------

### 类：InsertManager

**作用**：管理块引用(INSERT)实体和块定义(BLOCK_RECORD)，提供坐标变换和块展开功能。

#### 构造方法

```python
InsertManager(loader: DwgJsonLoader)
```

#### 块定义管理

- `get_all_block_definitions() -> dict[str, BlockDefinition]`：获取所有块定义。
- `get_block_definition(name: str) -> Optional[BlockDefinition]`：按名称获取块定义。
- `get_block_definition_by_handle(handle: int) -> Optional[BlockDefinition]`：按句柄获取块定义。
- `get_user_blocks() -> dict[str, BlockDefinition]`：获取用户定义的块（排除系统块和匿名块）。
- `get_anonymous_blocks() -> dict[str, BlockDefinition]`：获取匿名块（如 `*D5`, `*D7`）。
- `block_exists(name: str) -> bool`：检查块是否存在。
- `get_block_names() -> list[str]`：获取所有块名称。

#### INSERT实体管理

- `get_all_inserts(include_nested: bool = True) -> list[InsertInfo]`：获取所有INSERT实体。
- `get_inserts_by_block_name(block_name: str) -> list[InsertInfo]`：按块名获取INSERT。
- `get_inserts_by_layer(layer_name: str) -> list[InsertInfo]`：按图层获取INSERT。
- `get_inserts_by_layers(layer_names: list[str]) -> list[InsertInfo]`：按多个图层获取INSERT。

#### 坐标变换

- `transform_point(local_point: dict, insert_info: InsertInfo, consider_base_point: bool = True) -> TransformedPoint`： 将块内局部坐标变换为世界坐标。
- `transform_points(local_points: list[dict], insert_info: InsertInfo) -> list[TransformedPoint]`：批量变换点坐标。
- `get_transformation_matrix(insert_info: InsertInfo) -> list[list[float]]`：获取4x4变换矩阵。

#### 块展开

- `expand_insert(insert_info: InsertInfo, max_depth: int = 10, transform_coords: bool = True) -> list[dict]`： 递归展开INSERT，获取块内所有实体（可选变换坐标到世界坐标系）。

#### 块属性处理

- `get_insert_attributes(insert_info: InsertInfo) -> dict[str, str]`：获取INSERT的属性值（标签→值）。
- `get_block_attribute_definitions(block_name: str) -> list[dict]`：获取块定义中的ATTDEF。

#### 统计查询

- `count_inserts_by_block() -> dict[str, int]`：统计每个块被引用的次数。
- `get_block_usage_info() -> list[dict]`：获取块使用情况汇总。
- `find_unused_blocks() -> list[str]`：查找未被使用的块。
- `get_nesting_depth(block_name: str) -> int`：计算块的最大嵌套深度。

#### dataclass：BlockDefinition

```python
@dataclass
class BlockDefinition:
    name: str                    # 块名称
    handle: int                  # 块句柄
    base_point: dict[str, float] # 块基点坐标 {x, y, z}
    entities: list[dict]         # 块内实体列表
    insertion_units: int         # 插入单位
    explodability: bool          # 是否可分解
    scalability: bool            # 是否可缩放
    description: str             # 块描述
    is_anonymous: bool           # 是否为匿名块（以*开头）
    is_layout: bool              # 是否为布局块
    
    @property
    def entity_count() -> int                           # 块内实体数量
    def get_entities_by_type(type: str) -> list[dict]   # 获取块内指定类型实体
    def has_nested_inserts() -> bool                    # 是否包含嵌套INSERT
```

#### dataclass：InsertInfo

```python
@dataclass
class InsertInfo:
    handle: int                      # INSERT实体句柄
    block_name: str                  # 引用的块名称
    insertion_point: dict[str, float] # 插入点坐标 {x, y, z}
    x_scale: float                   # X方向缩放
    y_scale: float                   # Y方向缩放
    z_scale: float                   # Z方向缩放
    rotation: float                  # 旋转角度（弧度）
    layer: str                       # 所在图层
    column_count: int                # 阵列列数
    row_count: int                   # 阵列行数
    column_spacing: float            # 列间距
    row_spacing: float               # 行间距
    attributes: list[dict]           # 块属性列表
    raw_data: dict                   # 原始数据
    owner_block_handle: int          # 所属块的handle
    
    @property
    def is_array() -> bool           # 是否为阵列插入
    def has_uniform_scale() -> bool  # 是否为均匀缩放
    def is_mirrored() -> bool        # 是否镜像
```

#### dataclass：TransformedPoint

```python
@dataclass
class TransformedPoint:
    x: float
    y: float
    z: float
    original: Optional[dict]  # 原始坐标引用
    
    def to_dict() -> dict[str, float]
```

#### 使用示例

```python
from dwg_parser import DwgParser

parser = DwgParser("drawing.json")
insert_mgr = parser.insert_manager

# 获取所有用户定义的块
user_blocks = insert_mgr.get_user_blocks()
print(f"用户块: {list(user_blocks.keys())}")

# 获取指定块的所有引用
door_inserts = insert_mgr.get_inserts_by_block_name("Door")
for ins in door_inserts:
    print(f"门位置: {ins.insertion_point}, 旋转: {ins.rotation}")

# 坐标变换：将块内局部坐标转换为世界坐标
ins = door_inserts[0]
local_pt = {"x": 100, "y": 0, "z": 0}
world_pt = insert_mgr.transform_point(local_pt, ins)
print(f"局部坐标 {local_pt} → 世界坐标 ({world_pt.x:.2f}, {world_pt.y:.2f})")

# 展开块，获取块内所有实体（坐标已变换到世界坐标系）
expanded_entities = insert_mgr.expand_insert(ins, transform_coords=True)
for entity in expanded_entities:
    print(f"  {entity['type']}: {entity}")

# 统计块使用情况
counts = insert_mgr.count_inserts_by_block()
for name, count in counts.items():
    print(f"块 '{name}' 被引用 {count} 次")
```

------

### 主类：DwgParser

**作用**：高层封装，提供最常用的解析接口，整合所有管理器。

#### 构造方法

```python
DwgParser(
    json_path: Optional[str] = None,
    json_data: Optional[dict] = None,
    json_bytes: Optional[bytes] = None
)
```

#### 类方法

- `from_file(json_path: str) -> DwgParser`
- `from_dict(json_data: dict) -> DwgParser`
- `from_bytes(json_bytes: bytes) -> DwgParser`
- `from_response(response) -> DwgParser`

#### 属性

- `loader: DwgJsonLoader` - 数据加载器
- `layer_manager: LayerManager` - 图层管理器
- `entity_manager: EntityManager` - 实体管理器
- `insert_manager: InsertManager` - 块引用管理器

#### 图层相关方法

- `get_all_layer_names() -> list[str]`
- `match_layers(pattern: str) -> list[str]`

#### 实体相关方法

- `get_all_entities(include_nested: bool = True) -> list[EntityInfo]`
- `get_entities_by_type(entity_type: str) -> list[EntityInfo]`
- `get_entities_by_layers(layer_names: list[str]) -> list[EntityInfo]`
- `extract_entity_attrs(entity_info: EntityInfo) -> Optional[dict]`：提取标准化属性。
- `get_entity_attr(entity_info: EntityInfo, attr_name: str) -> Any`：获取单个属性值。
- `get_supported_entity_types() -> list[str]`：获取支持的实体类型列表。

#### INSERT相关便捷方法

- `get_all_block_definitions() -> dict[str, BlockDefinition]`
- `get_block_definition(name: str) -> Optional[BlockDefinition]`
- `get_all_inserts(include_nested: bool = True) -> list[InsertInfo]`

#### 使用示例

```python
from dwg_parser import DwgParser

# 创建解析器
parser = DwgParser("drawing.json")

# 图层操作
layers = parser.get_all_layer_names()
beam_layers = parser.match_layers(r"(?i).*BEAM.*")

# 实体操作
all_entities = parser.get_all_entities()
texts = parser.get_entities_by_type("TEXT")
for text in texts:
    attrs = parser.extract_entity_attrs(text)
    print(f"文字: {attrs['text_content']}, 位置: {attrs['position']}")

# INSERT操作
blocks = parser.get_all_block_definitions()
inserts = parser.get_all_inserts()
```

------

### 实体提取器

**作用**：为不同类型的实体提取标准化属性。

#### 已注册的提取器

- `TEXT` - 单行文字
- `MTEXT` - 多行文字
- `LINE` - 直线
- `CIRCLE` - 圆
- `ARC` - 圆弧
- `LWPOLYLINE` - 轻量多段线
- `INSERT` - 块引用

#### 使用方法

```python
from dwg_parser import get_extractor

# 获取提取器
extractor = get_extractor("TEXT")
attrs = extractor.extract(text_entity)

# 或使用 DwgParser 便捷方法
attrs = parser.extract_entity_attrs(entity)
```

#### INSERT提取器返回属性

```python
{
    # 通用属性
    "layer": str,              # 图层名
    "handle": int,             # 实体句柄
    "color": int,              # 颜色索引
    "is_visible": bool,        # 是否可见
    
    # INSERT特有属性
    "block_name": str,         # 引用的块名称
    "insertion_point": dict,   # 插入点 {x, y, z}
    "x_scale": float,          # X缩放
    "y_scale": float,          # Y缩放
    "z_scale": float,          # Z缩放
    "rotation": float,         # 旋转角度(弧度)
    "column_count": int,       # 阵列列数
    "row_count": int,          # 阵列行数
    "column_spacing": float,   # 列间距
    "row_spacing": float,      # 行间距
    "attributes": dict,        # 块属性 {标签: 值}
    "raw_attributes": list,    # 原始属性数据
    
    # 派生属性
    "is_array": bool,          # 是否为阵列
    "is_mirrored": bool,       # 是否镜像
    "has_uniform_scale": bool, # 是否均匀缩放
}
```

#### 其他实体提取器返回属性

**TEXT**

```python
{
    "text_content": str,    # 文字内容
    "position": dict,       # 位置 {x, y, z}
    "height": float,        # 文字高度
    "rotation": float,      # 旋转角度
    "x_scale": float,       # X方向缩放
    "bounding_box": dict,   # 包围盒
}
```

**LINE**

```python
{
    "start_point": dict,    # 起点 {x, y, z}
    "end_point": dict,      # 终点 {x, y, z}
    "length": float,        # 长度
    "angle": float,         # 角度(弧度)
    "bounding_box": dict,   # 包围盒
}
```

**CIRCLE**

```python
{
    "center": dict,         # 圆心 {x, y, z}
    "radius": float,        # 半径
    "bounding_box": dict,   # 包围盒
}
```

**ARC**

```python
{
    "center": dict,         # 圆心
    "radius": float,        # 半径
    "start_angle": float,   # 起始角度
    "end_angle": float,     # 终止角度
    "arc_length": float,    # 弧长
    "start_point": dict,    # 起点坐标
    "end_point": dict,      # 终点坐标
}
```

**LWPOLYLINE**

```python
{
    "vertices": list,       # 顶点列表 [{x, y, bulge}, ...]
    "vertex_count": int,    # 顶点数量
    "is_closed": bool,      # 是否闭合
    "bounding_box": dict,   # 包围盒
}
```

------

## 项目结构

```
ArchitecturalDrawingsProcessing/
├── dwg_parser/
│   ├── __init__.py           # 主模块，导出DwgParser等
│   ├── loader.py             # DwgJsonLoader
│   ├── entity_manager.py     # EntityManager, EntityInfo
│   ├── layer_manager.py      # LayerManager
│   ├── insert_manager.py     # InsertManager, BlockDefinition, InsertInfo
│   ├── utils.py              # 工具函数
│   └── entity_extractors/
│       ├── __init__.py       # 提取器注册
│       ├── base.py           # BaseEntityExtractor
│       ├── text.py           # TextExtractor
│       ├── mtext.py          # MTextExtractor
│       ├── line.py           # LineExtractor
│       ├── circle.py         # CircleExtractor
│       ├── arc.py            # ArcExtractor
│       ├── lwpolyline.py     # LwPolylineExtractor
│       └── insert.py         # InsertExtractor
├── consumer.py               # RabbitMQ消费者
├── app.py                    # Flask健康检查服务
├── test.py                   # 单元测试
├── demo.py                   # 功能演示
└── README.md
```

------

## 使用建议

1. **基础解析**：使用`DwgParser`快速提取实体和图层。
2. **块引用处理**：使用`InsertManager`管理块定义、坐标变换和块展开。
3. **属性提取**：使用`extract_entity_attrs()`获取标准化的实体属性。
4. **生产集成**：参考`consumer.py`使用RabbitMQ消费任务。

如需进一步扩展或有具体问题，可根据接口添加新实体提取器或匹配规则。