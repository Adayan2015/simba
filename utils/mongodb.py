from pymongo import MongoClient
from bson.objectid import ObjectId

from utils.config import MONGODB_SERVER, MONGODB_PORT, MONGODB_DB_NAME, MONGODB_COLLECTION_MESSAGE

client = MongoClient(MONGODB_SERVER, MONGODB_PORT)
db = client.get_database(MONGODB_DB_NAME)
# db.authenticate(MONGODB_USER, MONGODB_PWD)

ObjectId = ObjectId

if __name__ == '__main__':
    for i in range(1, 10):
        print(MONGODB_COLLECTION_MESSAGE)
        obj_id = db.get_collection(MONGODB_COLLECTION_MESSAGE).insert({'name': 'test'})
        print(obj_id)
