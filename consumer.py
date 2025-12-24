from rocketmq.client import PushConsumer, ConsumeStatus
import time, sys

def on_message(msg):
    print('[消费到] msgId=%s  body=%s' % (msg.id, msg.body.decode('utf-8')))
    return ConsumeStatus.CONSUME_SUCCESS   # 告诉 Broker 消费成功

consumer = PushConsumer('python-demo-group')  # 消费组名随意
consumer.set_namesrv_addr('localhost:9876')   # NameServer 地址
consumer.subscribe('TopicTest', on_message)   # 订阅主题（* 表示全部 Tag）
consumer.start()

print('[INFO] 消费者已启动，等待消息…  Ctrl+C 退出')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('正在关闭…')
    consumer.shutdown()