import io
from pymongo import MongoClient
from bson.json_util import dumps

"""
Copied data from mongodb to local client
"""


def mongo_input():
    # client = MongoClient()
    client = MongoClient("mongodb://lionking:Tv6pAzDp@60.205.187.223:27017/Simba?authMechanism=SCRAM-SHA-1")

    db = client.Simba
    a = db.faq.find({})

    with io.open('data.json', 'w', encoding='utf-8') as f:
        f.write(dumps(a, ensure_ascii=False))
        print('Backed up mongodb!')

# cursor = db.faq.find()
# for document in cursor:
#    print(document)
