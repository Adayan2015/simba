import time
from utils import mongodb


class ChatMessagePlatform:
    (WechatPerson, WechatGroup, WexinGongzhong) = ('chat_person', 'chat_group', 'gongzhong')


# voice itchat还不确定怎么发
class ChatMessageType:
    (Text, Picture, Voice, Video, File) = ('Text', 'Picture', 'Voice', 'Video', 'File')


class ChatMessageSender:
    (Customer, Bot) = ('customer', 'bot')


# 是否后台人工给客户
class ChatMessageSendAuto:
    (Yes, No) = ('1', '0')


# 对话库消息格式
chat_message_dict = {
    'platform': ChatMessagePlatform.WechatPerson,  # 入口类型：chat_person, chat_group
    'type': ChatMessageType.Text,  # 消息类型：Text, Picture, Voice
    'session_id': '',  # 会话标识，微信公众号：md5(openid)、itchat个人号：md5(fromUserName)、itchat群聊：md5(fromUserName)
    'session_uid': '',  # 会话用户标识，用于区分群聊里的哪个用户，微信公众号：md5(openid)、个人号：md5(fromUserName)、群聊：md5(ActualUserName)
    'session_name': '',  # 会话标题，微信公众号：公众号名称、个人号：用户名称、群聊：群组名
    'sender': ChatMessageSender.Customer,  # 发送人：customer、bot
    'auto': ChatMessageSendAuto.Yes,  # 1-自动发送，0－人工发送
    'customer_id': '',  # itchat用户ID, 用于回复时发送的地址：itchat个人号：fromUserName、itchat群聊：fromUserName
    'customer_showname': '',  # 用户显示名，group中发送时要@上
    'bot_id': '',  # 机器人ID
    'content': '',  #
    'desc': '',
    'request_body': '',
    'create_time': 0,
    'plansend_time': 0,  # 计划发送时间
    'send_time': 0,  # 发送时间
}

# 七鱼回调消息格式
message_qiyu_dict = {
    'session_uid': '',
    'msg_id': '',
    'msg_type': '',
    'content': '',
    'fetched': False,  # 是否已经取过
    'create_time': int(time.time())
}

# 多轮会话采集
message_multidialog_dict = {
    'session_uid': '',  # 用户身份
    'plan': '',  # 具体险种,
    'label': '',  # 套餐／单品／二轮,
    'switch': False,  # True/False（用以确定是否进入信息收集）,
    'info': '',  # 用户信息收集进程
    'waiting': ''  # 等待用户的信息, 什么结构？e.g. age, sex
}


# store message to mongo
def store_message(chat_message_dict):
    msg_object_id = mongodb.db.get_collection(mongodb.MONGODB_COLLECTION_MESSAGE).insert(chat_message_dict)
    return msg_object_id


def find_message(msg_object_id):
    msg_info = mongodb.db.get_collection(mongodb.MONGODB_COLLECTION_MESSAGE).find_one(
        {'_id': mongodb.ObjectId(msg_object_id)})
    return msg_info


# 更新消息内容
def update_message(msg_object_id, update_info_dict):
    allowKeys = ['send_time']
    new_update_info_dict = {}
    for key, value in update_info_dict.items():
        if key in allowKeys:
            new_update_info_dict[key] = value
    mongodb.db.get_collection(mongodb.MONGODB_COLLECTION_MESSAGE).update({'_id': mongodb.ObjectId(msg_object_id)},
                                                                         {'$set': new_update_info_dict})
    pass


if __name__ == '__main__':
    import time
    import copy

    print(ChatMessageType.Text)
    new_chat_message_dict = copy.deepcopy(chat_message_dict)  # chat_message_dict.copy()
    print(new_chat_message_dict)

    update_message('59634500c3666e08fd4f71f4', {'send_time': int(time.time())})
