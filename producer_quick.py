# producer_quick.py
from rocketmq.client import Producer, Message

p = Producer('quick-producer')
p.set_namesrv_addr('localhost:9876')
p.start()

msg = Message('TopicTest')
msg.set_keys('key1')
msg.set_tags('tagA')
msg.set_body('Hello from Python Producer')

ret = p.send_sync(msg)
print('send ok, msgId =', ret.msg_id)
p.shutdown()