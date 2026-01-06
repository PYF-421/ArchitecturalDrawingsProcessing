"""
demo.py - DWG Parser 功能演示

演示 dwg_parser 模块的全部功能，包括：
1. 数据加载 (DwgJsonLoader)
2. 图层管理 (LayerManager)
3. 实体管理 (EntityManager)
4. 块引用管理 (InsertManager)
5. 实体属性提取 (各类Extractor)

"""

import sys
import json
import math

sys.path.insert(0, '.')

from dwg_parser import (
    DwgParser,
    DwgJsonLoader,
    EntityInfo,
    InsertManager,
    InsertInfo,
    BlockDefinition,
    TransformedPoint,
    get_extractor,
    register_extractor,
    EXTRACTOR_REGISTRY,
)

JSON_PATH = r"G:\Desktop\test\test.json"

def print_header(title: str):
    """打印标题"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def print_subheader(title: str):
    """打印子标题"""
    print(f"\n--- {title} ---")


def format_point(point: dict, precision: int = 2) -> str:
    """格式化点坐标"""
    x = round(point.get('x', 0), precision)
    y = round(point.get('y', 0), precision)
    z = point.get('z', 0)
    if z != 0:
        return f"({x}, {y}, {round(z, precision)})"
    return f"({x}, {y})"


def demo_loader(json_path: str):
    """DwgJsonLoader"""
    print_header("1. DwgJsonLoader - 数据加载")
    
    # 方式1: 从文件路径加载
    print_subheader("1.1 从文件加载")
    loader = DwgJsonLoader(json_path=json_path)
    data = loader.load()
    print(f"loader = DwgJsonLoader(json_path='{json_path}')")
    print(f"data = loader.load()")
    print(f"  → 加载成功，数据类型: {type(data).__name__}")
    
    # 方式2: 从字典加载
    print_subheader("1.2 从字典加载")
    print("loader2 = DwgJsonLoader(json_data=existing_dict)")
    
    # 方式3: 从字节加载
    print_subheader("1.3 从字节加载")
    print("loader3 = DwgJsonLoader(json_bytes=json_bytes)")
    
    # 方式4: 从HTTP响应加载
    print_subheader("1.4 从HTTP响应加载")
    print("loader4 = DwgJsonLoader.from_response(response)")
    
    # 获取tables
    print_subheader("1.5 获取数据结构")
    tables = loader.get_tables()
    print(f"tables = loader.get_tables()")
    print(f"  → 包含的表: {list(tables.keys())}")
    
    # 获取块记录
    block_records = loader.get_block_records()
    print(f"\nblock_records = loader.get_block_records()")
    print(f"  → 块记录数量: {len(block_records)}")
    
    # 获取图层
    layers = loader.get_layers()
    print(f"\nlayers = loader.get_layers()")
    print(f"  → 图层数量: {len(layers)}")
    
    # 获取块映射
    block_map = loader.get_block_map()
    print(f"\nblock_map = loader.get_block_map()")
    print(f"  → 映射键数量: {len(block_map)}")
    
    # 按名称获取块
    print_subheader("1.6 按名称/句柄获取块")
    model_space = loader.get_block_by_name("*Model_Space")
    print(f"model_space = loader.get_block_by_name('*Model_Space')")
    print(f"  → 实体数量: {len(model_space.get('entities', []))}")
    
    return loader


def demo_layer_manager(parser: DwgParser):
    """LayerManager"""
    print_header("2. LayerManager - 图层管理")
    
    layer_mgr = parser.layer_manager
    
    # 获取所有图层名
    print_subheader("2.1 获取所有图层名")
    layer_names = layer_mgr.get_all_layer_names()
    print(f"layer_names = layer_mgr.get_all_layer_names()")
    print(f"  → 总数: {len(layer_names)}")
    print(f"  → 前5个: {layer_names[:5]}")
    
    # 正则匹配图层
    print_subheader("2.2 正则匹配图层")
    
    beam_layers = layer_mgr.match_layers(r"(?i).*BEAM.*")
    print(f"beam_layers = layer_mgr.match_layers(r'(?i).*BEAM.*')")
    print(f"  → 匹配数量: {len(beam_layers)}")
    print(f"  → 示例: {beam_layers[:3]}")
    
    axis_layers = layer_mgr.match_layers(r".*AXIS.*")
    print(f"\naxis_layers = layer_mgr.match_layers(r'.*AXIS.*')")
    print(f"  → 匹配数量: {len(axis_layers)}")
    
    # 完全匹配
    print_subheader("2.3 完全匹配图层")
    exact = layer_mgr.match_layers_fullmatch(r"0")
    print(f"exact = layer_mgr.match_layers_fullmatch(r'0')")
    print(f"  → 结果: {exact}")
    
    # 检查图层是否存在
    print_subheader("2.4 检查图层是否存在")
    exists = layer_mgr.layer_exists("0")
    print(f"layer_mgr.layer_exists('0') → {exists}")
    
    not_exists = layer_mgr.layer_exists("NOT_EXIST")
    print(f"layer_mgr.layer_exists('NOT_EXIST') → {not_exists}")
    
    # 获取图层信息
    print_subheader("2.5 获取图层详细信息")
    layer_info = layer_mgr.get_layer_info("0")
    if layer_info:
        print(f"layer_info = layer_mgr.get_layer_info('0')")
        print(f"  → name: {layer_info.get('name')}")
        print(f"  → color: {layer_info.get('color')}")
        print(f"  → frozen: {layer_info.get('frozen', False)}")


def demo_entity_manager(parser: DwgParser):
    """EntityManager"""
    print_header("3. EntityManager - 实体管理")
    
    entity_mgr = parser.entity_manager
    
    # 获取所有实体
    print_subheader("3.1 获取所有实体")
    all_entities = entity_mgr.get_all_entities(include_nested=True)
    print(f"all_entities = entity_mgr.get_all_entities(include_nested=True)")
    print(f"  → 总数(含嵌套): {len(all_entities)}")
    
    top_entities = entity_mgr.get_all_entities(include_nested=False)
    print(f"\ntop_entities = entity_mgr.get_all_entities(include_nested=False)")
    print(f"  → 顶层实体数: {len(top_entities)}")
    
    # 获取实体类型
    print_subheader("3.2 获取实体类型列表")
    entity_types = entity_mgr.get_entity_types()
    print(f"entity_types = entity_mgr.get_entity_types()")
    print(f"  → 类型: {entity_types}")
    
    # 按类型统计
    print_subheader("3.3 按类型统计实体数量")
    type_counts = entity_mgr.count_entities_by_type()
    print(f"type_counts = entity_mgr.count_entities_by_type()")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  → {t}: {count}")
    
    # 按类型获取实体
    print_subheader("3.4 按类型获取实体")
    lines = entity_mgr.get_entities_by_type("LINE")
    print(f"lines = entity_mgr.get_entities_by_type('LINE')")
    print(f"  → 数量: {len(lines)}")
    
    texts = entity_mgr.get_entities_by_type("TEXT")
    print(f"\ntexts = entity_mgr.get_entities_by_type('TEXT')")
    print(f"  → 数量: {len(texts)}")
    
    # 按图层获取实体
    print_subheader("3.5 按图层获取实体")
    layer_entities = entity_mgr.get_entities_by_layer(["0"])
    print(f"layer_entities = entity_mgr.get_entities_by_layer(['0'])")
    print(f"  → 数量: {len(layer_entities)}")
    
    # 按类型和图层获取
    print_subheader("3.6 按类型和图层组合过滤")
    filtered = entity_mgr.get_entities_by_type_and_layer("TEXT", ["BEAM-JZ-V"])
    print(f"filtered = entity_mgr.get_entities_by_type_and_layer('TEXT', ['BEAM-JZ-V'])")
    print(f"  → 数量: {len(filtered)}")
    
    # EntityInfo 属性演示
    print_subheader("3.7 EntityInfo 对象属性")
    if all_entities:
        entity = all_entities[0]
        print(f"entity = all_entities[0]")
        print(f"  → entity.entity_type: {entity.entity_type}")
        print(f"  → entity.handle: {entity.handle}")
        print(f"  → entity.layer: {entity.layer}")
        print(f"  → entity.block_path: {entity.block_path}")
        print(f"  → entity.owner_block: {entity.owner_block}")
        print(f"  → entity.is_in_model_space(): {entity.is_in_model_space()}")
        print(f"  → entity.get_nesting_depth(): {entity.get_nesting_depth()}")


def demo_insert_manager(parser: DwgParser):
    """InsertManager"""
    print_header("4. InsertManager - 块引用管理")
    
    insert_mgr = parser.insert_manager
    
    # 获取所有块定义
    print_subheader("4.1 获取所有块定义")
    block_defs = insert_mgr.get_all_block_definitions()
    print(f"block_defs = insert_mgr.get_all_block_definitions()")
    print(f"  → 总数: {len(block_defs)}")
    for name, block in list(block_defs.items())[:5]:
        print(f"     - {name}: {block.entity_count} entities")
    
    # 获取用户块和匿名块
    print_subheader("4.2 获取用户块 vs 匿名块")
    user_blocks = insert_mgr.get_user_blocks()
    anon_blocks = insert_mgr.get_anonymous_blocks()
    print(f"user_blocks = insert_mgr.get_user_blocks()")
    print(f"  → 用户块: {list(user_blocks.keys())}")
    print(f"\nanon_blocks = insert_mgr.get_anonymous_blocks()")
    print(f"  → 匿名块: {list(anon_blocks.keys())}")
    
    # 按名称获取块定义
    print_subheader("4.3 按名称/句柄获取块定义")
    if user_blocks:
        block_name = list(user_blocks.keys())[0]
        block = insert_mgr.get_block_definition(block_name)
        print(f"block = insert_mgr.get_block_definition('{block_name}')")
        print(f"  → handle: {block.handle}")
        print(f"  → base_point: {format_point(block.base_point)}")
        print(f"  → entity_count: {block.entity_count}")
        print(f"  → is_anonymous: {block.is_anonymous}")
        print(f"  → has_nested_inserts(): {block.has_nested_inserts()}")
    
    # BlockDefinition 方法
    print_subheader("4.4 BlockDefinition 方法")
    if user_blocks:
        block = list(user_blocks.values())[0]
        lines_in_block = block.get_entities_by_type("LINE")
        print(f"block.get_entities_by_type('LINE')")
        print(f"  → 块内LINE数量: {len(lines_in_block)}")
    
    # 获取所有INSERT实体
    print_subheader("4.5 获取所有INSERT实体")
    all_inserts = insert_mgr.get_all_inserts(include_nested=True)
    print(f"all_inserts = insert_mgr.get_all_inserts(include_nested=True)")
    print(f"  → 总数: {len(all_inserts)}")
    
    top_inserts = insert_mgr.get_all_inserts(include_nested=False)
    print(f"\ntop_inserts = insert_mgr.get_all_inserts(include_nested=False)")
    print(f"  → Model_Space中: {len(top_inserts)}")
    
    # InsertInfo 属性
    print_subheader("4.6 InsertInfo 对象属性")
    if all_inserts:
        ins = all_inserts[0]
        print(f"ins = all_inserts[0]")
        print(f"  → ins.handle: {ins.handle}")
        print(f"  → ins.block_name: {ins.block_name}")
        print(f"  → ins.insertion_point: {format_point(ins.insertion_point)}")
        print(f"  → ins.rotation: {ins.rotation:.4f} rad ({math.degrees(ins.rotation):.1f}°)")
        print(f"  → ins.x_scale: {ins.x_scale}")
        print(f"  → ins.y_scale: {ins.y_scale}")
        print(f"  → ins.is_array: {ins.is_array}")
        print(f"  → ins.is_mirrored: {ins.is_mirrored}")
        print(f"  → ins.has_uniform_scale: {ins.has_uniform_scale}")
    
    # 按块名/图层获取INSERT
    print_subheader("4.7 按块名/图层获取INSERT")
    if user_blocks:
        block_name = list(user_blocks.keys())[0]
        inserts_by_name = insert_mgr.get_inserts_by_block_name(block_name)
        print(f"insert_mgr.get_inserts_by_block_name('{block_name}')")
        print(f"  → 数量: {len(inserts_by_name)}")
    
    inserts_by_layer = insert_mgr.get_inserts_by_layer("0")
    print(f"\ninsert_mgr.get_inserts_by_layer('0')")
    print(f"  → 数量: {len(inserts_by_layer)}")
    
    # 坐标变换
    print_subheader("4.8 坐标变换 (局部坐标 → 世界坐标)")
    if all_inserts:
        ins = all_inserts[0]
        local_pt = {"x": 100, "y": 50, "z": 0}
        world_pt = insert_mgr.transform_point(local_pt, ins, consider_base_point=True)
        print(f"local_pt = {local_pt}")
        print(f"world_pt = insert_mgr.transform_point(local_pt, ins)")
        print(f"  → 世界坐标: {format_point(world_pt.to_dict())}")
        
        # 批量变换
        local_pts = [{"x": 0, "y": 0, "z": 0}, {"x": 100, "y": 0, "z": 0}]
        world_pts = insert_mgr.transform_points(local_pts, ins)
        print(f"\nworld_pts = insert_mgr.transform_points(local_pts, ins)")
        for i, (lp, wp) in enumerate(zip(local_pts, world_pts)):
            print(f"  → {format_point(lp)} → {format_point(wp.to_dict())}")
    
    # 变换矩阵
    print_subheader("4.9 获取4x4变换矩阵")
    if all_inserts:
        ins = all_inserts[0]
        matrix = insert_mgr.get_transformation_matrix(ins)
        print(f"matrix = insert_mgr.get_transformation_matrix(ins)")
        for row in matrix:
            formatted = [f"{v:8.4f}" for v in row]
            print(f"  [{', '.join(formatted)}]")
    
    # 块展开
    print_subheader("4.10 展开块 (获取块内所有实体)")
    for name, block in user_blocks.items():
        if block.entity_count > 0:
            inserts = insert_mgr.get_inserts_by_block_name(name)
            if inserts:
                # 不变换坐标
                expanded_raw = insert_mgr.expand_insert(inserts[0], transform_coords=False)
                print(f"expanded = insert_mgr.expand_insert(ins, transform_coords=False)")
                print(f"  → 块 '{name}' 展开后: {len(expanded_raw)} 个实体")
                
                # 变换坐标
                expanded_transformed = insert_mgr.expand_insert(inserts[0], transform_coords=True)
                print(f"\nexpanded = insert_mgr.expand_insert(ins, transform_coords=True)")
                print(f"  → 展开并变换: {len(expanded_transformed)} 个实体")
                
                # 显示展开的实体类型
                types = {}
                for e in expanded_raw:
                    t = e.get('type', 'UNKNOWN')
                    types[t] = types.get(t, 0) + 1
                print(f"  → 实体类型分布: {types}")
                break
    
    # 块属性
    print_subheader("4.11 块属性处理")
    if all_inserts:
        ins = all_inserts[0]
        attrs = insert_mgr.get_insert_attributes(ins)
        print(f"attrs = insert_mgr.get_insert_attributes(ins)")
        print(f"  → 属性: {attrs if attrs else '(无属性)'}")
        
        attdefs = insert_mgr.get_block_attribute_definitions(ins.block_name)
        print(f"\nattdefs = insert_mgr.get_block_attribute_definitions('{ins.block_name}')")
        print(f"  → ATTDEF数量: {len(attdefs)}")
    
    # 统计功能
    print_subheader("4.12 统计功能")
    counts = insert_mgr.count_inserts_by_block()
    print(f"counts = insert_mgr.count_inserts_by_block()")
    for name, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  → {name}: {count} 次引用")
    
    # 块使用信息
    print(f"\nusage_info = insert_mgr.get_block_usage_info()")
    usage_info = insert_mgr.get_block_usage_info()
    print(f"  → 返回 {len(usage_info)} 个块的详细使用信息")
    
    # 未使用的块
    unused = insert_mgr.find_unused_blocks()
    print(f"\nunused = insert_mgr.find_unused_blocks()")
    print(f"  → 未使用的块: {unused}")
    
    # 嵌套深度
    print_subheader("4.13 计算块嵌套深度")
    for name in list(user_blocks.keys())[:3]:
        depth = insert_mgr.get_nesting_depth(name)
        print(f"insert_mgr.get_nesting_depth('{name}') → {depth}")


def demo_extractors(parser: DwgParser):
    """实体提取器"""
    print_header("5. 实体属性提取器")
    
    # 显示已注册的提取器
    print_subheader("5.1 已注册的提取器")
    print(f"EXTRACTOR_REGISTRY.keys() → {list(EXTRACTOR_REGISTRY.keys())}")
    
    # TEXT 提取器
    print_subheader("5.2 TEXT 提取器")
    texts = parser.get_entities_by_type("TEXT")
    if texts:
        extractor = get_extractor("TEXT")
        attrs = extractor.extract(texts[0])
        print(f"extractor = get_extractor('TEXT')")
        print(f"attrs = extractor.extract(text_entity)")
        print(f"  → text_content: {attrs['text_content']}")
        print(f"  → position: {format_point(attrs['position'])}")
        print(f"  → height: {attrs['height']}")
        print(f"  → rotation: {attrs['rotation']:.4f} rad")
        print(f"  → bounding_box: {attrs['bounding_box']}")
    
    # MTEXT 提取器
    print_subheader("5.3 MTEXT 提取器")
    mtexts = parser.get_entities_by_type("MTEXT")
    if mtexts:
        extractor = get_extractor("MTEXT")
        attrs = extractor.extract(mtexts[0])
        print(f"attrs = get_extractor('MTEXT').extract(mtext_entity)")
        print(f"  → text_content: {attrs['text_content'][:30]}...")
        print(f"  → position: {format_point(attrs['position'])}")
        print(f"  → rect_width: {attrs['rect_width']:.2f}")
        print(f"  → rect_height: {attrs['rect_height']:.2f}")
    
    # LINE 提取器
    print_subheader("5.4 LINE 提取器")
    lines = parser.get_entities_by_type("LINE")
    if lines:
        extractor = get_extractor("LINE")
        attrs = extractor.extract(lines[0])
        print(f"attrs = get_extractor('LINE').extract(line_entity)")
        print(f"  → start_point: {format_point(attrs['start_point'])}")
        print(f"  → end_point: {format_point(attrs['end_point'])}")
        print(f"  → length: {attrs['length']:.2f}")
        print(f"  → angle: {attrs['angle']:.4f} rad ({math.degrees(attrs['angle']):.1f}°)")
    
    # CIRCLE 提取器
    print_subheader("5.5 CIRCLE 提取器")
    circles = parser.get_entities_by_type("CIRCLE")
    if circles:
        extractor = get_extractor("CIRCLE")
        attrs = extractor.extract(circles[0])
        print(f"attrs = get_extractor('CIRCLE').extract(circle_entity)")
        print(f"  → center: {format_point(attrs['center'])}")
        print(f"  → radius: {attrs['radius']:.2f}")
    
    # ARC 提取器
    print_subheader("5.6 ARC 提取器")
    arcs = parser.get_entities_by_type("ARC")
    if arcs:
        extractor = get_extractor("ARC")
        attrs = extractor.extract(arcs[0])
        print(f"attrs = get_extractor('ARC').extract(arc_entity)")
        print(f"  → center: {format_point(attrs['center'])}")
        print(f"  → radius: {attrs['radius']:.2f}")
        print(f"  → start_angle: {attrs['start_angle']:.4f} rad")
        print(f"  → end_angle: {attrs['end_angle']:.4f} rad")
        print(f"  → arc_length: {attrs['arc_length']:.2f}")
    
    # LWPOLYLINE 提取器
    print_subheader("5.7 LWPOLYLINE 提取器")
    polylines = parser.get_entities_by_type("LWPOLYLINE")
    if polylines:
        extractor = get_extractor("LWPOLYLINE")
        attrs = extractor.extract(polylines[0])
        print(f"attrs = get_extractor('LWPOLYLINE').extract(polyline_entity)")
        print(f"  → vertex_count: {attrs['vertex_count']}")
        print(f"  → is_closed: {attrs['is_closed']}")
        print(f"  → vertices (前2个):")
        for v in attrs['vertices'][:2]:
            print(f"       ({v['x']:.2f}, {v['y']:.2f}) bulge={v['bulge']}")
    
    # INSERT 提取器
    print_subheader("5.8 INSERT 提取器")
    inserts = parser.get_entities_by_type("INSERT")
    if inserts:
        extractor = get_extractor("INSERT")
        attrs = extractor.extract(inserts[0])
        print(f"attrs = get_extractor('INSERT').extract(insert_entity)")
        print(f"  → block_name: {attrs['block_name']}")
        print(f"  → insertion_point: {format_point(attrs['insertion_point'])}")
        print(f"  → x_scale: {attrs['x_scale']}")
        print(f"  → rotation: {attrs['rotation']:.4f} rad")
        print(f"  → is_array: {attrs['is_array']}")
        print(f"  → attributes: {attrs['attributes']}")
    
    # 使用 parser 便捷方法
    print_subheader("5.9 使用 DwgParser 便捷方法提取属性")
    if texts:
        attrs = parser.extract_entity_attrs(texts[0])
        print(f"attrs = parser.extract_entity_attrs(entity)")
        print(f"  → 返回完整属性字典")
        
        text_content = parser.get_entity_attr(texts[0], "text_content")
        print(f"\ntext = parser.get_entity_attr(entity, 'text_content')")
        print(f"  → {text_content}")


def demo_practical_examples(parser: DwgParser):
    """实际使用"""
    print_header("6. 实际应用示例")
    
    # 示例1: 提取所有梁标注文字
    print_subheader("6.1 提取梁图层的所有文字标注")
    beam_layers = parser.match_layers(r"(?i).*BEAM.*")
    print(f"# 查找梁相关图层")
    print(f"beam_layers = parser.match_layers(r'(?i).*BEAM.*')")
    print(f"  → 找到 {len(beam_layers)} 个图层")
    
    if beam_layers:
        beam_texts = parser.entity_manager.get_entities_by_type_and_layer("TEXT", beam_layers)
        print(f"\n# 获取这些图层上的TEXT实体")
        print(f"beam_texts = entity_mgr.get_entities_by_type_and_layer('TEXT', beam_layers)")
        print(f"  → 找到 {len(beam_texts)} 个文字标注")
        
        if beam_texts:
            print(f"\n# 提取文字内容")
            for text in beam_texts[:3]:
                attrs = parser.extract_entity_attrs(text)
                print(f"  → {attrs['text_content']}")
    
    # 示例2: 统计各类实体数量
    print_subheader("6.2 生成实体统计报告")
    print("# 统计实体数量")
    counts = parser.entity_manager.count_entities_by_type()
    total = sum(counts.values())
    print(f"总实体数: {total}")
    print("各类型占比:")
    for t, c in sorted(counts.items(), key=lambda x: -x[1]):
        pct = c / total * 100
        bar = '█' * int(pct / 5)
        print(f"  {t:12} {c:4} ({pct:5.1f}%) {bar}")
    
    # 示例3: 查找并展开特定块
    print_subheader("6.3 查找特定块并展开其内容")
    user_blocks = parser.insert_manager.get_user_blocks()
    for name, block in user_blocks.items():
        if block.has_nested_inserts():
            print(f"# 找到有嵌套的块: {name}")
            depth = parser.insert_manager.get_nesting_depth(name)
            print(f"  嵌套深度: {depth}")
            
            inserts = parser.insert_manager.get_inserts_by_block_name(name)
            if inserts:
                expanded = parser.insert_manager.expand_insert(inserts[0])
                print(f"  展开后实体数: {len(expanded)}")
                
                # 统计展开后的类型
                types = {}
                for e in expanded:
                    t = e.get('type', 'UNKNOWN')
                    types[t] = types.get(t, 0) + 1
                print(f"  包含: {types}")
            break
    
    # 示例4: 坐标范围分析
    print_subheader("6.4 分析图纸坐标范围")
    print("# 计算所有LINE实体的坐标范围")
    lines = parser.get_entities_by_type("LINE")
    if lines:
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for line in lines:
            attrs = parser.extract_entity_attrs(line)
            for pt in [attrs['start_point'], attrs['end_point']]:
                min_x = min(min_x, pt['x'])
                min_y = min(min_y, pt['y'])
                max_x = max(max_x, pt['x'])
                max_y = max(max_y, pt['y'])
        
        print(f"  X范围: {min_x:.2f} ~ {max_x:.2f}")
        print(f"  Y范围: {min_y:.2f} ~ {max_y:.2f}")
        print(f"  宽度: {max_x - min_x:.2f}")
        print(f"  高度: {max_y - min_y:.2f}")


def main():
    """主函数"""
    # 获取JSON文件路径
    json_path = JSON_PATH
    
    print("="*70)
    print(f"JSON文件: {json_path}")
    
    # 检查文件
    try:
        with open(json_path, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"\n错误: 找不到文件 '{json_path}'")
        print("用法: python demo.py [json_file_path]")
        sys.exit(1)
    
    # 创建解析器
    parser = DwgParser(json_path)
    
    # 运行各部分演示
    demo_loader(json_path)
    demo_layer_manager(parser)
    demo_entity_manager(parser)
    demo_insert_manager(parser)
    demo_extractors(parser)
    demo_practical_examples(parser)
    


if __name__ == "__main__":
    main()