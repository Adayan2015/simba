import json
import requests
from chatbots.base import BaseBot
from chatbots.config import get_config


class TulingBot(BaseBot):
    """docstring for TulingBot."""

    def __init__(self):
        self.bot_name = 'tuling'
        self.bot_config = get_config(self.bot_name)

    def request_text(self, msg):
        print('using TulingBot')
        apiKey = self.bot_config['tuling_apikey']
        apiUrl = self.bot_config['tuling_apiurl']

        try:
            data = {
                "reqType": 0,
                "perception": {
                    "inputText": {
                        "text": msg
                    },
                },
                "userInfo": {
                    "apiKey": apiKey,
                    "userId": "adayan"
                }
            }
            data_json = json.dumps(data)
            r = requests.post(apiUrl, data=data_json).json()
            replys = r['results'][0]['values']['text']
            return replys
        except:
            return


if __name__ == '__main__':
    a = TulingBot()
    msg = "你好"
    reply = a.request_text(msg)
    print(reply)
