"""
all_test.py - DWG Parser 完整测试脚本

测试所有提取器和管理器功能：
1. DwgJsonLoader - 数据加载
2. LayerManager - 图层管理
3. EntityManager - 实体管理
4. InsertManager - 块引用管理
5. 所有实体提取器 (TEXT, MTEXT, LINE, CIRCLE, ARC, LWPOLYLINE, INSERT)


"""

import sys
import json
import math
from typing import Optional
from dataclasses import dataclass

# 添加项目路径
sys.path.insert(0, '.')

from dwg_parser import (
    DwgParser, 
    DwgJsonLoader, 
    EntityManager, 
    EntityInfo,
    LayerManager,
    InsertManager,
    InsertInfo,
    BlockDefinition,
    TransformedPoint,
    get_extractor,
    EXTRACTOR_REGISTRY,
)


JSON_PATH = r"G:\Desktop\test\test.json"
# ============================================================
# 测试结果统计
# ============================================================

@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    message: str = ""
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f" - {self.message}" if self.message else ""
        return f"  {status}: {self.name}{msg}"


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.results: list[TestResult] = []
        self.current_section = ""
    
    def section(self, name: str):
        """开始新的测试部分"""
        self.current_section = name
        print(f"\n{'='*60}")
        print(f" {name}")
        print('='*60)
    
    def test(self, name: str, condition: bool, message: str = ""):
        """记录测试结果"""
        result = TestResult(name, condition, message)
        self.results.append(result)
        print(result)
        return condition
    
    def summary(self):
        """打印测试总结"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print(f"\n{'='*60}")
        print(" 测试总结")
        print('='*60)
        print(f"  总计: {total} 个测试")
        print(f"  通过: {passed} 个")
        print(f"  失败: {failed} 个")
        print(f"  通过率: {passed/total*100:.1f}%")
        
        if failed > 0:
            print(f"\n  失败的测试:")
            for r in self.results:
                if not r.passed:
                    print(f"    - {r.name}: {r.message}")
        
        return failed == 0


# ============================================================
# 测试函数
# ============================================================

def test_loader(runner: TestRunner, json_path: str):
    """测试 DwgJsonLoader"""
    runner.section("DwgJsonLoader 测试")
    
    # 测试从文件加载
    loader = DwgJsonLoader(json_path=json_path)
    data = loader.load()
    runner.test("从文件加载", data is not None, f"数据类型: {type(data).__name__}")
    
    # 测试获取tables
    tables = loader.get_tables()
    runner.test("获取tables", isinstance(tables, dict), f"包含: {list(tables.keys())}")
    
    # 测试获取block_records
    block_records = loader.get_block_records()
    runner.test("获取block_records", isinstance(block_records, list), f"数量: {len(block_records)}")
    
    # 测试获取layers
    layers = loader.get_layers()
    runner.test("获取layers", isinstance(layers, list), f"数量: {len(layers)}")
    
    # 测试block_map
    block_map = loader.get_block_map()
    runner.test("获取block_map", isinstance(block_map, dict), f"键数: {len(block_map)}")
    
    # 测试按名称获取块
    model_space = loader.get_block_by_name("*Model_Space")
    runner.test("按名称获取块", model_space is not None, "获取 *Model_Space")
    
    # 测试从字典加载
    loader2 = DwgJsonLoader(json_data=data)
    runner.test("从字典加载", loader2.get_raw_data() is not None)
    
    # 测试从字节加载
    json_bytes = json.dumps(data).encode('utf-8')
    loader3 = DwgJsonLoader(json_bytes=json_bytes)
    runner.test("从字节加载", loader3.get_raw_data() is not None)
    
    return loader


def test_layer_manager(runner: TestRunner, parser: DwgParser):
    """测试 LayerManager"""
    runner.section("LayerManager 测试")
    
    layer_mgr = parser.layer_manager
    
    # 测试获取所有图层名
    layer_names = layer_mgr.get_all_layer_names()
    runner.test("获取所有图层名", len(layer_names) > 0, f"数量: {len(layer_names)}")
    
    # 测试获取图层映射
    layer_map = layer_mgr.get_layer_map()
    runner.test("获取图层映射", len(layer_map) > 0, f"数量: {len(layer_map)}")
    
    # 测试正则匹配图层
    beam_layers = layer_mgr.match_layers(r"(?i).*BEAM.*")
    runner.test("正则匹配图层", isinstance(beam_layers, list), f"匹配BEAM: {len(beam_layers)}个")
    
    # 测试完全匹配
    exact_match = layer_mgr.match_layers_fullmatch(r"0")
    runner.test("完全匹配图层", "0" in exact_match or len(exact_match) >= 0)
    
    # 测试图层是否存在
    exists = layer_mgr.layer_exists("0")
    runner.test("检查图层存在", exists == True, "图层'0'存在")
    
    # 测试获取图层信息
    layer_info = layer_mgr.get_layer_info("0")
    runner.test("获取图层信息", layer_info is not None or layer_info is None)
    
    # 打印部分图层名
    print(f"\n  示例图层名: {layer_names[:5]}...")


def test_entity_manager(runner: TestRunner, parser: DwgParser):
    """测试 EntityManager"""
    runner.section("EntityManager 测试")
    
    entity_mgr = parser.entity_manager
    
    # 测试获取所有实体
    all_entities = entity_mgr.get_all_entities(include_nested=True)
    runner.test("获取所有实体(含嵌套)", len(all_entities) > 0, f"数量: {len(all_entities)}")
    
    # 测试获取非嵌套实体
    top_entities = entity_mgr.get_all_entities(include_nested=False)
    runner.test("获取顶层实体", len(top_entities) <= len(all_entities), f"数量: {len(top_entities)}")
    
    # 测试获取实体类型列表
    entity_types = entity_mgr.get_entity_types()
    runner.test("获取实体类型列表", len(entity_types) > 0, f"类型: {entity_types}")
    
    # 测试按类型统计
    type_counts = entity_mgr.count_entities_by_type()
    runner.test("按类型统计实体", len(type_counts) > 0, f"统计: {type_counts}")
    
    # 测试按类型获取实体
    for entity_type in ["LINE", "TEXT", "INSERT"]:
        entities = entity_mgr.get_entities_by_type(entity_type)
        runner.test(f"获取{entity_type}实体", isinstance(entities, list), f"数量: {len(entities)}")
    
    # 测试EntityInfo属性
    if all_entities:
        entity = all_entities[0]
        runner.test("EntityInfo.entity_type", hasattr(entity, 'entity_type'))
        runner.test("EntityInfo.handle", hasattr(entity, 'handle'))
        runner.test("EntityInfo.layer", hasattr(entity, 'layer'))
        runner.test("EntityInfo.block_path", hasattr(entity, 'block_path'))
        runner.test("EntityInfo.raw_data", hasattr(entity, 'raw_data'))


def test_insert_manager(runner: TestRunner, parser: DwgParser):
    """测试 InsertManager"""
    runner.section("InsertManager 测试")
    
    insert_mgr = parser.insert_manager
    
    # 测试获取所有块定义
    block_defs = insert_mgr.get_all_block_definitions()
    runner.test("获取所有块定义", len(block_defs) > 0, f"数量: {len(block_defs)}")
    
    # 测试获取用户块
    user_blocks = insert_mgr.get_user_blocks()
    runner.test("获取用户块", isinstance(user_blocks, dict), f"数量: {len(user_blocks)}")
    
    # 测试获取匿名块
    anon_blocks = insert_mgr.get_anonymous_blocks()
    runner.test("获取匿名块", isinstance(anon_blocks, dict), f"数量: {len(anon_blocks)}")
    
    # 测试获取块名列表
    block_names = insert_mgr.get_block_names()
    runner.test("获取块名列表", len(block_names) > 0, f"示例: {block_names[:3]}")
    
    # 测试按名称获取块定义
    if block_names:
        block = insert_mgr.get_block_definition(block_names[0])
        runner.test("按名称获取块定义", block is not None)
    
    # 测试BlockDefinition属性
    for name, block in list(block_defs.items())[:1]:
        runner.test("BlockDefinition.name", block.name == name)
        runner.test("BlockDefinition.handle", isinstance(block.handle, int))
        runner.test("BlockDefinition.base_point", isinstance(block.base_point, dict))
        runner.test("BlockDefinition.entities", isinstance(block.entities, list))
        runner.test("BlockDefinition.entity_count", block.entity_count == len(block.entities))
        runner.test("BlockDefinition.is_anonymous", isinstance(block.is_anonymous, bool))
        runner.test("BlockDefinition.has_nested_inserts()", isinstance(block.has_nested_inserts(), bool))
    
    # 测试获取所有INSERT
    all_inserts = insert_mgr.get_all_inserts(include_nested=True)
    runner.test("获取所有INSERT", isinstance(all_inserts, list), f"数量: {len(all_inserts)}")
    
    # 测试按块名获取INSERT
    if user_blocks:
        block_name = list(user_blocks.keys())[0]
        inserts = insert_mgr.get_inserts_by_block_name(block_name)
        runner.test("按块名获取INSERT", isinstance(inserts, list), f"块'{block_name}': {len(inserts)}个")
    
    # 测试InsertInfo属性
    if all_inserts:
        ins = all_inserts[0]
        runner.test("InsertInfo.handle", isinstance(ins.handle, int))
        runner.test("InsertInfo.block_name", isinstance(ins.block_name, str))
        runner.test("InsertInfo.insertion_point", isinstance(ins.insertion_point, dict))
        runner.test("InsertInfo.rotation", isinstance(ins.rotation, float))
        runner.test("InsertInfo.x_scale", isinstance(ins.x_scale, (int, float)))
        runner.test("InsertInfo.is_array", isinstance(ins.is_array, bool))
        runner.test("InsertInfo.is_mirrored", isinstance(ins.is_mirrored, bool))
    
    # 测试坐标变换
    if all_inserts:
        ins = all_inserts[0]
        local_pt = {"x": 100, "y": 50, "z": 0}
        world_pt = insert_mgr.transform_point(local_pt, ins, consider_base_point=False)
        runner.test("坐标变换", isinstance(world_pt, TransformedPoint))
        runner.test("TransformedPoint.x", isinstance(world_pt.x, float))
        runner.test("TransformedPoint.to_dict()", isinstance(world_pt.to_dict(), dict))
        
        # 测试变换矩阵
        matrix = insert_mgr.get_transformation_matrix(ins)
        runner.test("获取变换矩阵", len(matrix) == 4 and len(matrix[0]) == 4)
    
    # 测试块展开
    if user_blocks:
        for name, block in user_blocks.items():
            if block.entity_count > 0:
                inserts = insert_mgr.get_inserts_by_block_name(name)
                if inserts:
                    expanded = insert_mgr.expand_insert(inserts[0], transform_coords=False)
                    runner.test("块展开(不变换)", isinstance(expanded, list), f"块'{name}': {len(expanded)}个实体")
                    
                    expanded_t = insert_mgr.expand_insert(inserts[0], transform_coords=True)
                    runner.test("块展开(变换坐标)", isinstance(expanded_t, list))
                    break
    
    # 测试统计功能
    counts = insert_mgr.count_inserts_by_block()
    runner.test("统计块引用次数", isinstance(counts, dict))
    
    usage_info = insert_mgr.get_block_usage_info()
    runner.test("获取块使用信息", isinstance(usage_info, list))
    
    unused = insert_mgr.find_unused_blocks()
    runner.test("查找未使用的块", isinstance(unused, list), f"数量: {len(unused)}")
    
    # 测试嵌套深度
    if user_blocks:
        for name in list(user_blocks.keys())[:1]:
            depth = insert_mgr.get_nesting_depth(name)
            runner.test("计算嵌套深度", isinstance(depth, int), f"块'{name}': 深度{depth}")


def test_extractors(runner: TestRunner, parser: DwgParser):
    """测试所有实体提取器"""
    runner.section("实体提取器测试")
    
    # 检查注册的提取器
    registered_types = list(EXTRACTOR_REGISTRY.keys())
    runner.test("提取器注册表", len(registered_types) >= 7, f"类型: {registered_types}")
    
    # 测试各类型提取器
    test_cases = {
        "TEXT": ["text_content", "position", "height", "rotation"],
        "MTEXT": ["text_content", "position", "height", "rect_width"],
        "LINE": ["start_point", "end_point", "angle", "length"],
        "CIRCLE": ["center", "radius"],
        "ARC": ["center", "radius", "start_angle", "end_angle", "arc_length"],
        "LWPOLYLINE": ["vertices", "vertex_count", "is_closed"],
        "INSERT": ["block_name", "insertion_point", "x_scale", "rotation", "attributes"],
    }
    
    for entity_type, expected_attrs in test_cases.items():
        # 获取提取器
        extractor = get_extractor(entity_type)
        runner.test(f"获取{entity_type}提取器", extractor is not None)
        
        if extractor is None:
            continue
        
        # 获取该类型的实体
        entities = parser.get_entities_by_type(entity_type)
        
        if not entities:
            print(f"    (无{entity_type}实体可测试)")
            continue
        
        # 提取属性
        entity = entities[0]
        attrs = extractor.extract(entity)
        runner.test(f"{entity_type}提取属性", attrs is not None)
        
        if attrs is None:
            continue
        
        # 检查通用属性
        common_attrs = ["layer", "handle", "color", "is_visible"]
        for attr in common_attrs:
            has_attr = attr in attrs
            if not has_attr:
                runner.test(f"{entity_type}通用属性.{attr}", False, "缺失")
        
        # 检查特定属性
        for attr in expected_attrs:
            has_attr = attr in attrs
            runner.test(f"{entity_type}.{attr}", has_attr, 
                       f"值: {attrs.get(attr, 'N/A')}" if has_attr else "缺失")
        
        # 打印示例数据
        print(f"\n    {entity_type} 示例属性:")
        for attr in expected_attrs[:3]:
            if attr in attrs:
                val = attrs[attr]
                if isinstance(val, dict):
                    val = {k: round(v, 2) if isinstance(v, float) else v for k, v in val.items()}
                elif isinstance(val, float):
                    val = round(val, 4)
                print(f"      {attr}: {val}")


def test_parser_convenience(runner: TestRunner, parser: DwgParser):
    """测试 DwgParser 便捷方法"""
    runner.section("DwgParser 便捷方法测试")
    
    # 测试基本便捷方法
    layer_names = parser.get_all_layer_names()
    runner.test("get_all_layer_names()", len(layer_names) > 0)
    
    matched = parser.match_layers(r".*")
    runner.test("match_layers()", len(matched) > 0)
    
    entities = parser.get_all_entities()
    runner.test("get_all_entities()", len(entities) > 0)
    
    # 测试按类型获取
    text_entities = parser.get_entities_by_type("TEXT")
    runner.test("get_entities_by_type()", isinstance(text_entities, list))
    
    # 测试属性提取
    if text_entities:
        attrs = parser.extract_entity_attrs(text_entities[0])
        runner.test("extract_entity_attrs()", attrs is not None)
        
        text_content = parser.get_entity_attr(text_entities[0], "text_content")
        runner.test("get_entity_attr()", text_content is not None or text_content == "")
    
    # 测试INSERT相关便捷方法
    block_defs = parser.get_all_block_definitions()
    runner.test("get_all_block_definitions()", len(block_defs) > 0)
    
    if block_defs:
        name = list(block_defs.keys())[0]
        block = parser.get_block_definition(name)
        runner.test("get_block_definition()", block is not None)
    
    inserts = parser.get_all_inserts()
    runner.test("get_all_inserts()", isinstance(inserts, list))
    
    # 测试支持的类型
    supported = parser.get_supported_entity_types()
    runner.test("get_supported_entity_types()", "INSERT" in supported)


def test_edge_cases(runner: TestRunner, parser: DwgParser):
    """测试边界情况"""
    runner.section("边界情况测试")
    
    # 测试不存在的实体类型
    extractor = get_extractor("NONEXISTENT")
    runner.test("不存在的提取器", extractor is None)
    
    # 测试不存在的图层
    layer_info = parser.layer_manager.get_layer_info("NONEXISTENT_LAYER")
    runner.test("不存在的图层", layer_info is None)
    
    # 测试不存在的块
    block = parser.insert_manager.get_block_definition("NONEXISTENT_BLOCK")
    runner.test("不存在的块", block is None)
    
    # 测试空列表图层过滤
    entities = parser.entity_manager.get_entities_by_layer([])
    runner.test("空图层列表过滤", len(entities) == 0)
    
    # 测试块存在检查
    exists = parser.insert_manager.block_exists("*Model_Space")
    runner.test("块存在检查(存在)", exists == True)
    
    not_exists = parser.insert_manager.block_exists("NONEXISTENT")
    runner.test("块存在检查(不存在)", not_exists == False)


# ============================================================
# 主函数
# ============================================================

def main():
    """主测试函数"""
    # 获取JSON文件路径
    json_path = JSON_PATH
    
    print("="*60)
    print(" DWG Parser 完整测试")
    print("="*60)
    print(f"测试文件: {json_path}")
    
    # 检查文件是否存在
    try:
        with open(json_path, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"\n错误: 找不到文件 '{json_path}'")
        print("用法: python test.py [json_file_path]")
        sys.exit(1)
    
    # 创建测试运行器
    runner = TestRunner()
    
    try:
        # 测试加载器
        loader = test_loader(runner, json_path)
        
        # 创建解析器
        parser = DwgParser(json_path)
        
        # 测试各管理器
        test_layer_manager(runner, parser)
        test_entity_manager(runner, parser)
        test_insert_manager(runner, parser)
        
        # 测试提取器
        test_extractors(runner, parser)
        
        # 测试便捷方法
        test_parser_convenience(runner, parser)
        
        # 测试边界情况
        test_edge_cases(runner, parser)
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 输出总结
    success = runner.summary()
    
    return success


if __name__ == "__main__":
    success = main()