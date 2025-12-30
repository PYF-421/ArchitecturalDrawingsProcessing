import pika
import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()

JAVA_HOST = os.getenv('JAVA_HOST')
RABBIT_HOST = os.getenv('RABBIT_HOST')

def get_json(body: bytes) -> dict:
    """
    从字节数据中提取JSON对象
    """
    return json.loads(body.decode('utf-8'))

# 消费逻辑方法
def callback(ch, method, properties, body):
    """
    body: bytes -> JSON
    """
    raw_data = get_json(body)
    params = {
        "paperId": raw_data.get("paperId"),
        "fileType": "BIN"
    }
    resp = requests.get(f"{JAVA_HOST}paper/fetchJsonFile", params=params, stream=True)

    content_bytes = resp.content

    # 这个就是后端返回给python端的json数据
    json_data = json.loads(content_bytes.decode("utf-8"))
    """
    TODO 下面的部分去调用算法侧的方法最后使用Java端回调接口返回数据
    """

    # 手动确认
    ch.basic_ack(delivery_tag=method.delivery_tag)


def consume():
    # 1. 建立连接
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host='RABBIT_HOST',
            credentials=pika.PlainCredentials('admin', 'password')
        )
    )
    channel = connection.channel()

    # 2. 声明队列
    channel.queue_declare(queue='parse_paper', durable=True)

    # 3. 一次只处理一条
    channel.basic_qos(prefetch_count=1)

    # 4. 开始消费
    channel.basic_consume(
        queue='parse_paper',
        on_message_callback=callback,
        auto_ack=False
    )

    print("等待消息中...")
    channel.start_consuming()


if __name__ == "__main__":
    consume()

