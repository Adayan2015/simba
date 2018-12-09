from urllib.request import urlopen
from urllib.request import Request
from urllib.parse import urlencode
from urllib.request import HTTPError
import json
import jieba.posseg
import jieba.analyse
import re
from utils.LTML import LTML
from utils import mongodb
from chatbots import message
from chatbots.base import BaseBot


class TripleBot(BaseBot):
    """docstring for DialogBot."""

    def __init__(self):
        super(TripleBot, self).__init__()

    def find_index(self, s1, s2, ss):
        a = None
        b = None
        for i in range(len(ss)):
            if ss[i]["cont"] == s1:
                a = i
            elif ss[i]["cont"] == s2:
                b = i
        return a, b

    def my_str(self, s):
        if isinstance(s, list):
            if isinstance(s[0], list):
                ss = ""
                for every_list in s:
                    for i in range(len(every_list)):
                        for key, value in every_list[i].items():
                            ss += "%s: %s\t" % (key, value)
                    ss += '\n'
                return ss
            if isinstance(s[0], dict):
                ss = ""
                for i in range(len(s)):
                    for key, value in s[i].items():
                        ss += "%s: %s\n" % (key, value)
                return ss
            else:

                return " ".join(s)
        else:
            return s

    def parsing(self, segment):
        # 语法依存分析
        segment_utf = []
        for word, flag in segment:
            segment_utf.append((word, flag))

        ltml = LTML()
        ltml.build_from_words(segment_utf)
        xml = ltml.tostring()

        uri_base = "http://api.ltp-cloud.com/analysis/?"  # from HIT
        api_key = "U1d737t0u3ooUDibTtXgFxbsGo9YVI8RQoatDkqy"

        data = {
            # "api_key": "p1D280Q923TgJsxCHdDVUseO9eYurYzusZzD6UeS",  # from HIT
            "api_key": "U1d737t0u3ooUDibTtXgFxbsGo9YVI8RQoatDkqy",  # from USTC
            "text": xml,
            "format": "json",
            "pattern": "dp",
            "xml_input": "true"
        }
        params = urlencode(data)
        try:
            request = Request(uri_base)
            response = urlopen(request, params)
            content = response.read().strip()
            # print content
            return json.loads(content)[0][0]
        except HTTPError as ee:
            print(ee.reason)

    # 查找文档中的关键词
    def find_key(self, sentence):
        try:
            """
            提取文档的关键词，使用方法：jieba.analyse.extract_tag
            topK:提取几个关键词
            """

            tag_list = jieba.analyse.extract_tags(sentence, topK=2, withWeight=False, allowPOS=("nz", "n", "ns"))
            if len(tag_list) >= 2:
                name = tag_list[0]
                prop = tag_list[1]
            elif len(tag_list) == 1:
                name = tag_list[0]
                prop = None
            else:
                name = None
                prop = None
            return name, prop
        except:
            return None, None

    def search(self, name, prop):
        print("检索名字：" + str(name))

        try:
            answer = "产品无此属性"
            cursor = mongodb.db.get_collection('entity_all').find_one({"name": name})

            if cursor is None:
                answer = "查无此项"
            else:
                for i in range(len(cursor["props"])):
                    if (cursor["props"][i]["name"] == prop) or (prop in cursor["props"][i]["alias"]):
                        answer = self.my_str(cursor["props"][i]["value"])
                        break
        except:
            return "查无此项"

        return answer

    """
    @param content, {'type':, 'content':}
    @return session_uid, [respContent]
    查询数据库进行判断
    """

    def request(self, session_uid, mixcontent):

        print('using TripleBot')

        if 'type' not in mixcontent:
            return session_uid, None

        if mixcontent['type'] != 'Text':
            return session_uid, None

        if 'content' not in mixcontent:
            return session_uid, None

        name, prop = self.find_key(mixcontent["content"])
        try:
            if (prop is not None) and (len(prop) > len(name)):
                a = name
                name = prop
                prop = a
        except:
            return session_uid, None

        answer = self.search(name, prop)

        if answer == "查无此项":
            return session_uid, None
        elif re.match(r".*(.doc|.pdf|.docx)$", answer):
            respContent = [{"type": message.ChatMessageType.File, "content": "http:" + s.lstrip("http:")} for s in
                           answer.split(",")]
            return session_uid, respContent
        else:
            respContent = {"type": message.ChatMessageType.Text, "content": answer}
            return session_uid, [respContent]

    def fetch(self, session_uid):
        pass


if __name__ == '__main__':
    bot = TripleBot()

    """
    测试用例：
    content：
    你好，请问太平福利全佑的承保公司是哪个？
    嗨！你跟我说说泰康鑫瑞人生这款保险都有哪些保障权益？
    我想咨询一下康怡保两全保险相较其他产品的产品特点有哪些？
    我想知道华夏医保通的保险期间有多久?
    你晓得泰顺自驾保的保期有多久嘛？
    哈罗！你知道畅享世家这款产品的承保年龄吗？
    请告诉我飞常保这款产品属于哪种产品类型？   （未回答）
    友邦全佑至珍重疾保险计划的险种是什么？
    朋友啊我有个问题，传家保的支付方式是怎样的呢？ （未回答）
    我想了解一下学子无忧卡的缴费方式是怎么样的？
    我想买一份平安e家保，它的保费是多少啊？
    我对安馨定期寿险很感兴趣，它的价格是多少？ （未回答）
    平安意外险的保险条款？
    你好!请问大特保吃货险的支付方式是怎样的？ (未回答)
    请问社保的保障权益都有哪些？（未回答）
    
    """

    mixcontent = {'type': message.ChatMessageType.Text, 'content': '你好!请问大特保吃货险的理赔流程是怎样的？'}
    respSessuid, respListContent = bot.request('session_uid', mixcontent)
    if respListContent is not None:
        print(mixcontent["content"] + "\n" + str(respListContent[0]['content']) + "\n")
    else:
        print("抱歉，小小哒，不能解决您的问题幺！")