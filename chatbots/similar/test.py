from multiprocessing import Pool
import os
import time
import pymongo

list1 = {}
print('******')
list2 = [10, 4, 20, 2, 80, 57, 598, 563, 31, 57, 8, 89, 4631, 15699, 78, 13654, 32878, 696]


def mycallback(key_value):
    key_i = key_value[0]
    value_i = key_value[1]
    print('callback key is %s value is %d' % (key_i, value_i))
    list1[str(key_i)] = value_i


def long_time_task(name, input_j, compare_i):
    print('Run task %s (%s)...') % (name, os.getpid())
    print(input_j - compare_i)
    start_time = time.time()
    # time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.') % (name, (end - start_time))
    return name, (input_j - compare_i)


def single(j, input_j, num_list2):
    key_i = j
    value_i = input_j - num_list2
    print('callback key is %s value is %d') % (key_i, value_i)
    list1[str(key_i)] = value_i


if __name__ == '__main__':

    print('Parent process %s.' % os.getpid())
    print('input a number')
    input_i = int(input())
    start = time.time()
    p = Pool(8)
    for i in range(len(list2)):
        p.apply_async(long_time_task, args=(i, input_i, list2[i],), callback=mycallback)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(list1)
    print(time.time() - start)
    # for i in range(len(list2)):
    #     single(i,input_i,list2[2])
    # end = time.time()
    # print end-start
    client_local = pymongo.MongoClient('127.0.0.1', port=27017)
    db_local = client_local.get_database('Simba')
    collection_local = db_local.get_collection('faq')
    stra = collection_local.find_one({"question": '你好'})
    print(stra)

    client_dtb = pymongo.MongoClient(
        'mongodb://lionking:Tv6pAzDp@''60.205.187.223'':27017/Simba?authMechanism=SCRAM-SHA-1')
    db_dtb = client_dtb.get_database('Simba')
    collection_dtb = db_dtb.get_collection('profession')
    for item in collection_dtb.find():
        collection_local.insert(item)
    print('ok')
