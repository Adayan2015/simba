import itchat
import re
from chatbots.multidialog.test_multi import MultiDialogBot
from chatbots.tuling import TulingBot
from chatbots.similar.test_similar import SimilarBot
from chatbots.triple import TripleBot
from utils import nlp
from chatbots import message
from chatbots.config import distinguish_words

print(distinguish_words)
keys = ['088faba46ffe42bd9cd444bfb56f164c', '151ef94e012647fa9f59938215858672']

KEY = keys[1]

tuling = TulingBot()
similar = SimilarBot()
triple = TripleBot()
multi = MultiDialogBot()
last_answers = []

'''
策略1：构建分类器
策略2：基于词区分
'''


# 根据用户问题选择机器人回答
def choose_bot(msg):
    flag = 0

    # 添加值在slot_information中
    multi.fillSlotAll(msg)

    # 查看是否包含slot_information
    for value in multi.slot_information.values():
        if value != '':
            print("slot_information为：")
            print(multi.slot_information)
            if value != 'man' or value != 'woman':
                return 3

    # 查看是否包含保险相关的词
    for word in distinguish_words:
        if (word in msg) or ('险' in msg) or ('保' in msg):
            return 1

    return flag


@itchat.msg_register(itchat.content.TEXT)
def weixin_input(msg):
    msg = msg['Content']
    print("用户提问： " + msg)

    global last_answers
    bot_id = choose_bot(msg)
    print('choose bot %s' % bot_id)

    pattern = re.compile('[0-9]')
    reply = "抱歉，小小哒，不能帮您解决，请换个方式提问幺！"

    # 判断用户是否输入想要的机器人
    if re.findall(pattern, msg) != [] and last_answers != []:
        if int(re.findall(pattern, msg)[0]) <= 3:
            reply = last_answers[int(re.findall(pattern, msg)[0]) - 1]
            last_answers = []
    else:
        """
        0号代表图灵机器人，
        1号代表三元组检索机器人，
        2号代表些相似度检索机器人,
        3号代表多轮对话机器人
        
        目前采用的方式是先判断是否为图灵机器人回答的问题，
        如果不是先用三元组检索，如果无答案再用相似度检索机器人
        """
        if bot_id == 0:
            reply = tuling.request_text(msg)
            print("tuling机器人回复：" + str(reply))

        if bot_id == 3:
            reply = multi.process(msg)

        if bot_id == 1:
            mixcontent = {'type': message.ChatMessageType.Text, 'content': msg}
            respSessuid, respListContent = triple.request('session_uid', mixcontent)
            if respListContent is not None:
                reply = respListContent[0]['content']
                print("三元组检索机器人回答： " + reply)
            else:
                # 相似度检索机器人
                questions, answers, scores = similar.output(msg)

                if isinstance(questions, list):
                    if float(scores[0]) < 0.75 or scores[0] == 'nan':
                        reply = tuling.request_text(msg)
                    else:
                        last_answers = answers
                        for i in range(len(questions)):
                            questions[i] = str(i + 1) + questions[i]
                        reply = '您的问题是下面这几个中的哪个呢,请回复编号\n' + ('\n'.join(questions))
                if isinstance(questions, str):
                    if float(scores) < 0.8 or scores[0] == 'nan':
                        reply = tuling.request_text(msg)
                    else:
                        reply = answers

    if '' not in multi.slot_information.values():
        for key in multi.slot_information.keys():
            multi.slot_information[key] = ''

    return reply


if __name__ == '__main__':
    itchat.auto_login()
    # itchat.login()
    itchat.run()
