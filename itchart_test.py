import itchat
from itchat.content import *


@itchat.msg_register([PICTURE, TEXT])
def simple_reply(msg):
    if msg['Type'] == TEXT:
        ReplyContent = 'I received message: ' + msg['Content']
    if msg['Type'] == PICTURE:
        ReplyContent = 'I received picture: ' + msg['FileName']
    itchat.send_msg('nice to meet you', msg['FromUserName'])


itchat.auto_login()
itchat.run()
