from pymongo import MongoClient

"""
Copied data from mongodb to local client
"""


def mongo_update(data):
    client = MongoClient()
    # client = MongoClient("mongodb://lionking:Tv6pAzDp@60.205.187.223:27017/Simba?authMechanism=SCRAM-SHA-1")
    db = client.Simba

    for cur in data:
        _id = cur[0]
        new_label = cur[1]

        db.faq.update(
            {'_id': _id},
            {
                '$set': {
                    'status': new_label
                }
            }, upsert=False)

        print('Updated one!')
