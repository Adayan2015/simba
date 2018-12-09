from chatbots.base import BaseBot
from utils import nlp
from utils import config
from utils import mongodb
from chatbots import message


class ProfessionBot(BaseBot):
    def __init__(self):
        super(ProfessionBot, self).__init__()

    def is_profession_question(self, content):
        keywords = ['职业类别', '职业', '类别']
        for key in keywords:
            if key in content:
                return True
        return False

    def request(self, session_uid, mixContent, mixWords=None):
        if mixContent['type'] != 'Text':
            return session_uid, None
        if 'content' not in mixContent:
            return session_uid, None

        # 准入条件
        if not self.is_profession_question(mixContent['content']):
            return session_uid, None

        # 包含的职业名词
        stop_property = ['v', 'uj', 'm', 'o', 'e', 'y', 'zg', 'r']
        words_need = []
        for (word, flag) in mixWords:
            if flag not in stop_property and word not in ['职业类别', '职业', '类别', '人']:
                words_need.append(word)

        collection = mongodb.db.get_collection(config.MONGODB_COLLECTION_PROFESSION)
        contents = []
        for item in collection.find({'company': '太保财'}):
            str1 = item['大分类']
            str2 = item['中分类']
            str3 = item['小分类']
            str4 = item['职业类别']
            answer = str1 + '/' + str2 + '/' + str3 + ' ' + str4
            if self.judegment(words_need, answer):
                contents.append(answer)

        respListContent = None
        if len(contents) > 0:
            respListContent = [{'type': message.ChatMessageType.Text, 'content': '\n'.join(contents)}]

        return session_uid, respListContent

    # 判断是否答案之一
    def judegment(self, words, answer):
        for word in words:
            if word.strip() not in answer:
                return False
        return True

    def output(self, mixContent):
        mixWords = nlp.cut(mixContent['content'])
        session_uid, respListContent = self.request('session_uid', mixContent, mixWords)
        try:
            if respListContent != None:
                category = respListContent[0]['content'].split(' ')[-1]
                return int(str(category))
            else:
                return 3
        except:
            return 3


if __name__ == '__main__':
    mixListContent = [
        {"type": "Text", "content": "锯木工人是什么类别啊"},
        {"type": "Text", "content": "装瓶工人是什么类别"},
        {"type": "Text", "content": "高空作业 职业类别"},
        {"type": "Text", "content": "防暴警察职业"},
        {"type": "Text", "content": "我想问你全家桶，岳父岳母可以保么"},
        {"type": "Text", "content": "农民"},
    ]
    bot = ProfessionBot()

    for mixContent in mixListContent:
        print('Q:', mixContent['content'])
        print(bot.output(mixContent))
    # mixContent = {"type": "Text","content": "锯木工人是什么类别啊"}
    # bot = ProfessionBot()
