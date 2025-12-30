import pika

# 1. 建立连接
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='localhost',
        credentials=pika.PlainCredentials('admin', 'password')
    )
)

channel = connection.channel()

# 2. 声明队列
channel.queue_declare(queue='task_post', durable=True)

def callback(ch, method, properties, body):
    print("收到消息:", body.decode())

    # 手动确认
    ch.basic_ack(delivery_tag=method.delivery_tag)

# 3. 一次只处理一条
channel.basic_qos(prefetch_count=1)

# 4. 开始消费
channel.basic_consume(
    queue='task_post',
    on_message_callback=callback,
    auto_ack=False
)

print("等待消息中...")
channel.start_consuming()
