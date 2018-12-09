# !usr/bin/python
# -*- coding:utf-8 -*-

import os

# mongodb
env = os.environ.get('SIMBA')
print("env SIMBA=%s" % env)

MONGODB_DB_NAME = 'simba'
MONGODB_SERVER = '127.0.0.1'
MONGODB_PORT = 27017

# 所有collection定义
MONGODB_COLLECTION_MESSAGE = 'message'
MONGODB_COLLECTION_MESSAGE_QIYU = 'message_qiyu'
MONGODB_COLLECTION_BOT = 'bot'
MONGODB_COLLECTION_PROFESSION = 'profession'
MONGODB_COLLECTION_ENTITY = 'entity_all'
MONGODB_COLLECTION_POSTER = 'image_text'
MONGODB_COLLECTION_MULTIDIALOG = 'multidialog'
MONGODB_COLLECTION_PLAN = 'recommendation_plan'

# redis
REDIS_SERVER = '127.0.0.1'
REDIS_PORT = 6379
