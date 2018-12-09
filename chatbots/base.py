from chatbots.config import get_config


class BaseBot(object):
    '''机器人名, 用于区分人格配置'''
    bot_name = ''
    bot_config = {}

    """docstring for BaseBot."""

    def __init__(self):
        self.bot_name = 'basebot'
        self.bot_config = get_config(self.bot_name)

    '''请求机器人处理'''

    def request_text(self, msg):
        api_key = self.bot_config[0]
        api_url = self.bot_config[1]
        return

    # def set_botid(self, bot_id):
    #     # 获取mongo数据
    #     collection = mongodb.db.get_collection(mongodb.MONGODB_COLLECTION_BOT)
    #     bot_info = collection.find_one({'_id': mongodb.ObjectId(bot_id)})
    #     print bot_info
    #     if not bot_info:
    #         logging.getLogger('app').error('bot empty: %s' % bot_id)
    #         return
    #
    #     self.bot_id = bot_id
    #     self.bot_name = bot_info['name']
    #     self.bot_config = get_config(self.bot_name)


if __name__ == '__main__':
    bot = BaseBot()
    print(bot.bot_name)
