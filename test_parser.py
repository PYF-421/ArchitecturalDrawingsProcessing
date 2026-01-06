"""
DWG Parser 测试脚本

验证各模块功能是否正常工作

"""

from dwg_parser import DwgParser, get_extractor, register_extractor
from dwg_parser.entity_extractors import EXTRACTOR_REGISTRY
import json


def main():
    # 初始化解析器
    print("=" * 60)
    print("DWG Parser 功能测试")
    print("=" * 60)
    
    parser = DwgParser(r"G:\Desktop\test\地下室.json")
    
    # ==================== 测试图层管理功能 ====================
    print("\n【功能2测试：图层管理】")
    print("-" * 40)
    
    # 获取所有图层名
    all_layers = parser.get_all_layer_names()
    print(f"总共有 {len(all_layers)} 个图层")
    print(f"前10个图层: {all_layers[:10]}")
    
    # 正则匹配图层
    beam_layers = parser.match_layers(r"BEAM.*")
    print(f"\n匹配 'BEAM.*' 的图层: {beam_layers}")
    
    axis_layers = parser.match_layers(r".*AXIS.*")
    print(f"匹配 '.*AXIS.*' 的图层: {axis_layers}")
    
    # ==================== 测试实体管理功能 ====================
    print("\n【功能1测试：实体管理】")
    print("-" * 40)
    
    # 获取所有实体
    all_entities = parser.get_all_entities()
    print(f"总共有 {len(all_entities)} 个实体")
    
    # 统计各类型实体数量
    type_counts = parser.entity_manager.count_entities_by_type()
    print(f"实体类型统计: {type_counts}")
    
    # 获取支持的实体类型
    supported_types = parser.get_supported_entity_types()
    print(f"当前支持的实体类型: {supported_types}")
    
    # 展示块嵌套关系
    print("\n块嵌套关系示例:")
    shown = 0
    for entity in all_entities:
        if len(entity.block_path) > 1:  # 在嵌套块中
            print(f"  类型: {entity.entity_type}, 图层: {entity.layer}")
            print(f"  块路径: {' -> '.join(entity.block_path)}")
            print(f"  直接所属块: {entity.owner_block}")
            shown += 1
            if shown >= 3:
                break
    
    # ==================== 测试实体属性提取功能 ====================
    print("\n【功能3测试：实体属性提取】")
    print("-" * 40)
    
    # TEXT
    print("\n--- TEXT实体 ---")
    text_entities = parser.get_entities_by_type("TEXT")
    print(f"TEXT实体数量: {len(text_entities)}")
    if text_entities:
        attrs = parser.extract_entity_attrs(text_entities[0])
        print(f"第一个TEXT属性:")
        print(f"  文字内容: {attrs['text_content']}")
        print(f"  位置: ({attrs['position']['x']:.2f}, {attrs['position']['y']:.2f})")
        print(f"  高度: {attrs['height']}")
        print(f"  旋转角度: {attrs['rotation']:.4f} rad")
        print(f"  图层: {attrs['layer']}")
        print(f"  块路径: {attrs['block_path']}")
        print(f"  颜色索引: {attrs['color']}")
    
    # MTEXT
    print("\n--- MTEXT实体 ---")
    mtext_entities = parser.get_entities_by_type("MTEXT")
    print(f"MTEXT实体数量: {len(mtext_entities)}")
    if mtext_entities:
        attrs = parser.extract_entity_attrs(mtext_entities[0])
        print(f"第一个MTEXT属性:")
        print(f"  文字内容: {attrs['text_content']}")
        print(f"  位置: ({attrs['position']['x']:.2f}, {attrs['position']['y']:.2f})")
        print(f"  高度: {attrs['height']}")
        print(f"  旋转角度: {attrs['rotation']:.4f} rad")
    
    # LINE
    print("\n--- LINE实体 ---")
    line_entities = parser.get_entities_by_type("LINE")
    print(f"LINE实体数量: {len(line_entities)}")
    if line_entities:
        attrs = parser.extract_entity_attrs(line_entities[0])
        print(f"第一个LINE属性:")
        print(f"  起点: ({attrs['start_point']['x']:.2f}, {attrs['start_point']['y']:.2f})")
        print(f"  终点: ({attrs['end_point']['x']:.2f}, {attrs['end_point']['y']:.2f})")
        print(f"  角度: {attrs['angle']:.4f} rad")
        print(f"  长度: {attrs['length']:.2f}")
        print(f"  图层: {attrs['layer']}")
    
    # CIRCLE
    print("\n--- CIRCLE实体 ---")
    circle_entities = parser.get_entities_by_type("CIRCLE")
    print(f"CIRCLE实体数量: {len(circle_entities)}")
    if circle_entities:
        attrs = parser.extract_entity_attrs(circle_entities[0])
        print(f"第一个CIRCLE属性:")
        print(f"  圆心: ({attrs['center']['x']:.2f}, {attrs['center']['y']:.2f})")
        print(f"  半径: {attrs['radius']:.2f}")
        print(f"  包围盒: min({attrs['bounding_box']['min']['x']:.2f}, {attrs['bounding_box']['min']['y']:.2f})")
        print(f"          max({attrs['bounding_box']['max']['x']:.2f}, {attrs['bounding_box']['max']['y']:.2f})")
    
    # ARC
    print("\n--- ARC实体 ---")
    arc_entities = parser.get_entities_by_type("ARC")
    print(f"ARC实体数量: {len(arc_entities)}")
    if arc_entities:
        attrs = parser.extract_entity_attrs(arc_entities[0])
        print(f"第一个ARC属性:")
        print(f"  起点: ({attrs['start_point']['x']:.2f}, {attrs['start_point']['y']:.2f})")
        print(f"  终点: ({attrs['end_point']['x']:.2f}, {attrs['end_point']['y']:.2f})")
        print(f"  圆心: ({attrs['center']['x']:.2f}, {attrs['center']['y']:.2f})")
        print(f"  半径: {attrs['radius']:.2f}")
        print(f"  起始角度: {attrs['start_angle']:.4f} rad")
        print(f"  终止角度: {attrs['end_angle']:.4f} rad")
        print(f"  总角度: {attrs['total_angle']:.4f} rad")
        print(f"  弧长: {attrs['arc_length']:.2f}")
    
    # LWPOLYLINE
    print("\n--- LWPOLYLINE实体 ---")
    polyline_entities = parser.get_entities_by_type("LWPOLYLINE")
    print(f"LWPOLYLINE实体数量: {len(polyline_entities)}")
    if polyline_entities:
        attrs = parser.extract_entity_attrs(polyline_entities[0])
        print(f"第一个LWPOLYLINE属性:")
        print(f"  顶点数量: {attrs['vertex_count']}")
        print(f"  是否闭合: {attrs['is_closed']}")
        print(f"  全局宽度: {attrs['constant_width']}")
        print(f"  起始宽度: {attrs['start_width']}")
        print(f"  终止宽度: {attrs['end_width']}")
        if attrs['vertices']:
            print(f"  前3个顶点:")
            for v in attrs['vertices'][:3]:
                print(f"    ({v['x']:.2f}, {v['y']:.2f}), bulge={v['bulge']}")
    
    # ==================== 测试功能4：按图层筛选实体 ====================
    print("\n【功能4测试：按图层筛选实体】")
    print("-" * 40)
    
    # 使用正则匹配获取图层，然后筛选实体
    beam_layers = parser.match_layers(r"BEAM.*")
    print(f"匹配 'BEAM.*' 的图层: {beam_layers}")
    
    if beam_layers:
        beam_entities = parser.get_entities_by_layers(beam_layers)
        print(f"这些图层中的实体数量: {len(beam_entities)}")
        
        # 统计这些实体的类型
        type_counts = {}
        for e in beam_entities:
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
        print(f"实体类型分布: {type_counts}")
        
        # 获取TEXT实体的文字内容
        text_in_beam = [e for e in beam_entities if e.entity_type == "TEXT"]
        if text_in_beam:
            print(f"\n这些图层中的TEXT内容（前5个）:")
            for e in text_in_beam[:5]:
                attrs = parser.extract_entity_attrs(e)
                print(f"  [{attrs['layer']}] {attrs['text_content']}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()