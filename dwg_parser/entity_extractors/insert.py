"""
INSERT实体属性提取器
"""

from typing import Any
from dwg_parser.entity_extractors.base import BaseEntityExtractor
from dwg_parser.utils import point_to_dict


class InsertExtractor(BaseEntityExtractor):
    """
    INSERT实体属性提取器
    
    INSERT是块引用实体，表示对块定义的一次引用/插入
    
    提取属性：
    - block_name: 引用的块定义名称
    - insertion_point: 插入点坐标
    - x_scale, y_scale, z_scale: X/Y/Z方向缩放比例
    - rotation: 旋转角度（弧度）
    - column_count, row_count: 阵列的列数/行数
    - column_spacing, row_spacing: 阵列的列间距/行间距
    - attributes: 块属性字典 {标签: 值}
    - raw_attributes: 原始属性数据列表
    - is_array: 是否为阵列插入
    - is_mirrored: 是否镜像（任一方向缩放为负）
    - has_uniform_scale: 是否为均匀缩放
    - 以及所有通用属性
    """
    
    ENTITY_TYPE = "INSERT"
    
    def extract(self, entity_info: Any) -> dict:
        """
        提取INSERT实体属性
        
        Args:
            entity_info: 实体信息对象
            
        Returns:
            属性字典
        """
        common = self._extract_common_attrs(entity_info)
        raw = entity_info.raw_data
        
        # 插入点
        insertion_point = point_to_dict(raw.get("insertionPoint"))
        
        # 缩放
        x_scale = raw.get("xScale", 1.0)
        y_scale = raw.get("yScale", 1.0)
        z_scale = raw.get("zScale", 1.0)
        
        # 阵列参数
        column_count = raw.get("columnCount", 0)
        row_count = raw.get("rowCount", 0)
        column_spacing = raw.get("columnSpacing", 0.0)
        row_spacing = raw.get("rowSpacing", 0.0)
        
        # 处理块属性(ATTRIB)
        raw_attribs = raw.get("attribs", [])
        attributes = self._parse_attributes(raw_attribs)
        
        # 计算派生属性
        is_array = column_count > 1 or row_count > 1
        is_mirrored = x_scale < 0 or y_scale < 0
        has_uniform_scale = (
            abs(x_scale - y_scale) < 1e-6 and 
            abs(y_scale - z_scale) < 1e-6
        )
        
        return {
            **common,
            # INSERT特有属性
            "block_name": raw.get("name", ""),
            "insertion_point": insertion_point,
            "x_scale": x_scale,
            "y_scale": y_scale,
            "z_scale": z_scale,
            "rotation": raw.get("rotation", 0.0),
            "column_count": column_count,
            "row_count": row_count,
            "column_spacing": column_spacing,
            "row_spacing": row_spacing,
            "attributes": attributes,
            "raw_attributes": raw_attribs,
            # 派生属性
            "is_array": is_array,
            "is_mirrored": is_mirrored,
            "has_uniform_scale": has_uniform_scale,
            # 拉伸方向
            "extrusion_direction": point_to_dict(raw.get("extrusionDirection")),
        }
    
    def _parse_attributes(self, raw_attribs: list) -> dict:
        """
        解析块属性列表
        
        Args:
            raw_attribs: 原始属性数据列表
            
        Returns:
            属性字典 {标签: 值}
        """
        attributes = {}
        for attr in raw_attribs:
            tag = attr.get("tag", "")
            # 属性值可能在text或value字段
            value = attr.get("text", "") or attr.get("value", "")
            if tag:
                attributes[tag] = value
        return attributes
    
    def _get_bounding_box(self, entity_info: Any) -> dict:
        """
        获取INSERT的包围盒
        
        注意：这只是插入点的位置，完整的包围盒需要考虑块内实体
        """
        raw = entity_info.raw_data
        insertion_point = point_to_dict(raw.get("insertionPoint"))
        return {
            "min": {"x": insertion_point["x"], "y": insertion_point["y"]},
            "max": {"x": insertion_point["x"], "y": insertion_point["y"]}
        }