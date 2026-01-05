"""
工具函数模块

提供几何计算和通用工具函数
"""

import math
from typing import Optional


def calculate_line_angle(start: dict, end: dict) -> float:
    """
    计算直线角度（弧度）
    
    Args:
        start: 起点坐标 {"x": float, "y": float}
        end: 终点坐标 {"x": float, "y": float}
        
    Returns:
        角度值（弧度），范围 [-π, π]
    """
    dx = end.get("x", 0) - start.get("x", 0)
    dy = end.get("y", 0) - start.get("y", 0)
    return math.atan2(dy, dx)


def calculate_line_length(start: dict, end: dict) -> float:
    """
    计算直线长度
    
    Args:
        start: 起点坐标 {"x": float, "y": float}
        end: 终点坐标 {"x": float, "y": float}
        
    Returns:
        直线长度
    """
    dx = end.get("x", 0) - start.get("x", 0)
    dy = end.get("y", 0) - start.get("y", 0)
    dz = end.get("z", 0) - start.get("z", 0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_arc_endpoint(
    center: dict, 
    radius: float, 
    angle: float
) -> dict:
    """
    根据圆心、半径和角度计算圆弧上的点
    
    Args:
        center: 圆心坐标 {"x": float, "y": float, "z": float}
        radius: 半径
        angle: 角度（弧度）
        
    Returns:
        点坐标 {"x": float, "y": float, "z": float}
    """
    return {
        "x": center.get("x", 0) + radius * math.cos(angle),
        "y": center.get("y", 0) + radius * math.sin(angle),
        "z": center.get("z", 0)
    }


def calculate_arc_endpoints(
    center: dict, 
    radius: float, 
    start_angle: float, 
    end_angle: float
) -> tuple[dict, dict]:
    """
    根据圆心、半径和起止角度计算圆弧起止点
    
    Args:
        center: 圆心坐标
        radius: 半径
        start_angle: 起始角度（弧度）
        end_angle: 终止角度（弧度）
        
    Returns:
        (起点坐标, 终点坐标)
    """
    start_point = calculate_arc_endpoint(center, radius, start_angle)
    end_point = calculate_arc_endpoint(center, radius, end_angle)
    return start_point, end_point


def calculate_arc_total_angle(start_angle: float, end_angle: float) -> float:
    """
    计算圆弧总角度（弧度）
    
    处理跨越0度的情况
    
    Args:
        start_angle: 起始角度（弧度）
        end_angle: 终止角度（弧度）
        
    Returns:
        总角度（弧度），始终为正值
    """
    total = end_angle - start_angle
    if total < 0:
        total += 2 * math.pi
    return total


def calculate_arc_length(radius: float, start_angle: float, end_angle: float) -> float:
    """
    计算弧长
    
    Args:
        radius: 半径
        start_angle: 起始角度（弧度）
        end_angle: 终止角度（弧度）
        
    Returns:
        弧长
    """
    total_angle = calculate_arc_total_angle(start_angle, end_angle)
    return radius * total_angle


def is_polyline_closed(flag: int) -> bool:
    """
    判断多段线是否闭合
    
    LWPOLYLINE的flag字段：
    - bit 0 (值1): 闭合
    - bit 7 (值128): 有凸起
    - bit 9 (值512): 等宽
    
    Args:
        flag: 多段线的flag值
        
    Returns:
        是否闭合
    """
    return bool(flag & 1)


def normalize_angle(angle: float) -> float:
    """
    将角度归一化到 [0, 2π) 范围
    
    Args:
        angle: 角度（弧度）
        
    Returns:
        归一化后的角度
    """
    two_pi = 2 * math.pi
    while angle < 0:
        angle += two_pi
    while angle >= two_pi:
        angle -= two_pi
    return angle


def radians_to_degrees(radians: float) -> float:
    """
    弧度转角度
    
    Args:
        radians: 弧度值
        
    Returns:
        角度值
    """
    return math.degrees(radians)


def degrees_to_radians(degrees: float) -> float:
    """
    角度转弧度
    
    Args:
        degrees: 角度值
        
    Returns:
        弧度值
    """
    return math.radians(degrees)


def calculate_bounding_box_line(start: dict, end: dict) -> dict:
    """
    计算直线的包围盒
    
    Args:
        start: 起点坐标
        end: 终点坐标
        
    Returns:
        包围盒 {"min": {"x", "y"}, "max": {"x", "y"}}
    """
    return {
        "min": {
            "x": min(start.get("x", 0), end.get("x", 0)),
            "y": min(start.get("y", 0), end.get("y", 0))
        },
        "max": {
            "x": max(start.get("x", 0), end.get("x", 0)),
            "y": max(start.get("y", 0), end.get("y", 0))
        }
    }


def calculate_bounding_box_circle(center: dict, radius: float) -> dict:
    """
    计算圆的包围盒
    
    Args:
        center: 圆心坐标
        radius: 半径
        
    Returns:
        包围盒 {"min": {"x", "y"}, "max": {"x", "y"}}
    """
    cx = center.get("x", 0)
    cy = center.get("y", 0)
    return {
        "min": {"x": cx - radius, "y": cy - radius},
        "max": {"x": cx + radius, "y": cy + radius}
    }


def calculate_bounding_box_arc(
    center: dict, 
    radius: float, 
    start_angle: float, 
    end_angle: float
) -> dict:
    """
    计算圆弧的包围盒
    
    需要考虑圆弧是否经过四个方向的极值点
    
    Args:
        center: 圆心坐标
        radius: 半径
        start_angle: 起始角度（弧度）
        end_angle: 终止角度（弧度）
        
    Returns:
        包围盒 {"min": {"x", "y"}, "max": {"x", "y"}}
    """
    cx = center.get("x", 0)
    cy = center.get("y", 0)
    
    # 起止点
    start_x = cx + radius * math.cos(start_angle)
    start_y = cy + radius * math.sin(start_angle)
    end_x = cx + radius * math.cos(end_angle)
    end_y = cy + radius * math.sin(end_angle)
    
    min_x = min(start_x, end_x)
    max_x = max(start_x, end_x)
    min_y = min(start_y, end_y)
    max_y = max(start_y, end_y)
    
    # 归一化角度
    start_angle = normalize_angle(start_angle)
    end_angle = normalize_angle(end_angle)
    
    # 检查是否经过四个方向的极值点
    def angle_in_arc(angle: float) -> bool:
        """检查角度是否在圆弧范围内"""
        angle = normalize_angle(angle)
        if start_angle <= end_angle:
            return start_angle <= angle <= end_angle
        else:  # 跨越0度
            return angle >= start_angle or angle <= end_angle
    
    # 右 (0度)
    if angle_in_arc(0):
        max_x = cx + radius
    # 上 (π/2)
    if angle_in_arc(math.pi / 2):
        max_y = cy + radius
    # 左 (π)
    if angle_in_arc(math.pi):
        min_x = cx - radius
    # 下 (3π/2)
    if angle_in_arc(3 * math.pi / 2):
        min_y = cy - radius
    
    return {
        "min": {"x": min_x, "y": min_y},
        "max": {"x": max_x, "y": max_y}
    }


def calculate_bounding_box_points(points: list[dict]) -> dict:
    """
    计算一组点的包围盒
    
    Args:
        points: 点列表，每个点为 {"x": float, "y": float}
        
    Returns:
        包围盒 {"min": {"x", "y"}, "max": {"x", "y"}}
    """
    if not points:
        return {"min": {"x": 0, "y": 0}, "max": {"x": 0, "y": 0}}
    
    xs = [p.get("x", 0) for p in points]
    ys = [p.get("y", 0) for p in points]
    
    return {
        "min": {"x": min(xs), "y": min(ys)},
        "max": {"x": max(xs), "y": max(ys)}
    }


def calculate_bounding_box_text(
    position: dict, 
    height: float, 
    text_content: str,
    rotation: float = 0,
    x_scale: float = 1.0
) -> dict:
    """
    估算文字的包围盒
    
    由于没有精确的字体信息，这里做简化估算
    假设平均字符宽度约为高度的0.6倍
    
    Args:
        position: 文字位置（左下角）
        height: 文字高度
        text_content: 文字内容
        rotation: 旋转角度（弧度）
        x_scale: X方向缩放
        
    Returns:
        包围盒 {"min": {"x", "y"}, "max": {"x", "y"}}
    """
    # 估算文字宽度（假设每个字符宽度为高度的0.6倍）
    char_width = height * 0.6 * x_scale
    text_width = len(text_content) * char_width
    
    px = position.get("x", 0)
    py = position.get("y", 0)
    
    if rotation == 0:
        return {
            "min": {"x": px, "y": py},
            "max": {"x": px + text_width, "y": py + height}
        }
    
    # 考虑旋转的情况，计算四个角点
    corners = [
        (0, 0),
        (text_width, 0),
        (text_width, height),
        (0, height)
    ]
    
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    
    rotated_corners = []
    for dx, dy in corners:
        rx = px + dx * cos_r - dy * sin_r
        ry = py + dx * sin_r + dy * cos_r
        rotated_corners.append({"x": rx, "y": ry})
    
    return calculate_bounding_box_points(rotated_corners)


def point_to_dict(point: Optional[dict]) -> dict:
    """
    标准化点坐标格式
    
    Args:
        point: 点坐标字典或None
        
    Returns:
        标准化的点坐标 {"x": float, "y": float, "z": float}
    """
    if point is None:
        return {"x": 0, "y": 0, "z": 0}
    return {
        "x": point.get("x", 0),
        "y": point.get("y", 0),
        "z": point.get("z", 0)
    }
