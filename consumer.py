"""
consumer_example.py - 消费者脚本示例

展示如何在RabbitMQ消费者中使用DwgParser处理JSON数据
"""

import pika
import json
import requests
from dotenv import load_dotenv
import os

# 导入解析器
from dwg_parser import DwgParser

load_dotenv()

JAVA_HOST = os.getenv('JAVA_HOST')
RABBIT_HOST = os.getenv('RABBIT_HOST')


def get_json(body: bytes) -> dict:
    """从字节数据中提取JSON对象"""
    return json.loads(body.decode('utf-8'))


def process_cad_data(parser: DwgParser, paper_id: str):
    """
    处理CAD数据的主要逻辑
    
    Args:
        parser: DwgParser实例
        paper_id: 图纸ID
    """
    # 获取所有图层
    layers = parser.get_all_layer_names()
    print(f"图层数量: {len(layers)}")
    
    # 获取所有实体
    entities = parser.get_all_entities()
    print(f"实体数量: {len(entities)}")
    
    # 统计实体类型
    type_counts = parser.entity_manager.count_entities_by_type()
    print(f"实体类型统计: {type_counts}")
    
    # 查找特定图层的实体
    beam_layers = parser.match_layers(r"(?i).*BEAM.*")
    if beam_layers:
        beam_entities = parser.get_entities_by_layers(beam_layers)
        print(f"梁图层实体数: {len(beam_entities)}")
    
    # 提取TEXT实体属性
    text_entities = parser.get_entities_by_type("TEXT")
    for text_entity in text_entities[:5]:  # 只处理前5个
        attrs = parser.extract_entity_attrs(text_entity)
        if attrs:
            print(f"TEXT: {attrs['text_content']}, 位置: ({attrs['position']['x']:.2f}, {attrs['position']['y']:.2f})")
    
    # TODO: 返回处理结果给Java端
    result = {
        "paperId": paper_id,
        "layerCount": len(layers),
        "entityCount": len(entities),
        "typeStats": type_counts
    }
    
    return result


def callback(ch, method, properties, body):
    """
    消息回调处理函数
    """
    raw_data = get_json(body)
    paper_id = raw_data.get("paperId")
    
    print(f"处理图纸: {paper_id}")
    
    # 从Java端获取JSON数据
    params = {
        "paperId": paper_id,
        "fileType": "BIN"
    }
    resp = requests.get(f"{JAVA_HOST}paper/fetchJsonFile", params=params, stream=True)
    
    # ============================================================
    # 方式1: 直接使用response对象
    # ============================================================
    parser = DwgParser.from_response(resp)
    
    # ============================================================
    # 方式2: 使用字节数据
    # ============================================================
    # parser = DwgParser(json_bytes=resp.content)
    
    # ============================================================
    # 方式3: 先解析为字典再加载
    # ============================================================
    # json_data = json.loads(resp.content.decode("utf-8"))
    # parser = DwgParser(json_data=json_data)
    
    # 处理CAD数据
    result = process_cad_data(parser, paper_id)
    
    # TODO: 使用Java端回调接口返回数据
    # callback_url = f"{JAVA_HOST}paper/callback"
    # requests.post(callback_url, json=result)
    
    # 手动确认消息
    ch.basic_ack(delivery_tag=method.delivery_tag)


def consume():
    """启动消费者"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBIT_HOST,
            credentials=pika.PlainCredentials('admin', 'password')
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue='parse_paper', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue='parse_paper',
        on_message_callback=callback,
        auto_ack=False
    )
    
    print("等待消息中...")
    channel.start_consuming()


if __name__ == "__main__":
    consume()