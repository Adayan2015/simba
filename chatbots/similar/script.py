import logging
from gensim.models import word2vec
import pymongo
import jieba.posseg as pseg
import re
import jieba

with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().split('\n')
jieba.load_userdict('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\entitydict.txt')

'''
连接mongo获取数据
'''


def mongo():
    client = pymongo.MongoClient('mongodb://lionking:Tv6pAzDp@''60.205.187.223'':27017/Simba?authMechanism=SCRAM-SHA-1')
    db = client.get_database('Simba')
    collection = db.get_collection('faq')
    pattern = re.compile('\n|\\&quot')
    q_a = []
    for item in collection.find():
        eachQA = []
        question = re.sub(pattern, '', item['question'])
        eachQA.append(question)
        answer = re.sub(pattern, '', item['answer'][0])
        # eachQA.append(question)
        eachQA.append(answer)
        q_a.append(eachQA)
    print('ok')
    for k in range(len(q_a)):
        print(q_a[k][0])
        print(q_a[k][1])
    with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\q&a_6W.txt', 'a', encoding='utf-8') as ff:
        for j in range(len(q_a)):
            ff.write(q_a[j][0])
            ff.write('\n')
            ff.write(q_a[j][1])
            ff.write('\n')
    print('finish')


'''
训练word2vec模型
'''
# 主程序
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 加载语料
sentences = word2vec.Text8Corpus("D:\similar\all_words_7_24.txt")
# 训练skip-gram模型; 默认window=5 size为生成向量大小，min_count为最低词频，alpha为学习率
models = word2vec.Word2Vec(sentences, size=200, min_count=3, min_alpha=0.025)
# 保存模型，以便重用
models.save(r"D:\PyCharm\PyCharmProjects\simba\chatbots\similar\datebao_no_eng_7_24.model")

'''
加载word2vec模型及测试
'''


def test_word2vec():
    model_s = word2vec.Word2Vec.load('D:\PyCharm\PyCharmProjects\simba\chatbots\similar\datebao_no_eng.model')
    y2 = model_s.most_similar(u"重疾险", topn=100)  # 20个最相关的
    print("和重疾险最相关的词有：\n")
    for item in y2:
        print(item[0], item[1])
    print("--------\n")
    y1 = model_s.similarity(u"重疾险", u"重大疾病保险")
    print("woman和man的相似度为：", y1)
    print("--------\n")
    # models[u"重疾险"]


'''
获得q_a
'''
with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\q&a.txt', 'r', encoding='utf-8') as f:
    q_a_1 = f.read().split('\n')
q = []
a = []
for i in range(len(q_a_1)):
    if i % 2 == 0:
        q.append(q_a_1[i])
    else:
        a.append(q_a_1[i])
q_a = []
for i in range(len(q)):
    q_a.append([q[i], a[i]])

'''
分词加生成各种文件
'''
with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().split('\n')
jieba.load_userdict('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\entitydict.txt')

all_words = []
q_all_words = []
a_all_words = []
count = 1
with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\q&a.txt', 'r', encoding='uft-8') as f:
    lines = f.read().split('\n')
    for line in lines:
        print(line)
        seg_list = pseg.cut(line.strip())
        list1 = []
        if count % 2 != 0:
            for word, flag in seg_list:
                if word.encode('utf-8') not in stop_words and word != ' ':
                    all_words.append(word.encode('utf-8'))
                    list1.append(word.encode('utf-8') + '_' + flag)
            q_all_words.append(list1)
        if count % 2 == 0:
            for word, flag in seg_list:
                if word not in stop_words and word != ' ':
                    all_words.append(word)
                    list1.append(word + '_' + flag)
            a_all_words.append(list1)
        count += 1

print(count, len(q_all_words), len(a_all_words))
print(a_all_words[0])
print(q_all_words[0])

with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\sep_questions_7_24.txt', 'a', encoding='utf-8') as f:
    for line in q_all_words:
        f.write(' '.join(line))
        f.write('\n')

with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\sep_answers.txt', 'a', encoding='utf-8') as f:
    for line in a_all_words:
        f.write(' '.join(line))
        f.write('\n')
print('finish')

with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\questions.txt', 'a', encoding='utf-8') as f:
    for i in range(len(q_a)):
        f.write(q_a[i][0])
        f.write('\n')

with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\answers.txt', 'a', encoding='utf-8') as f:
    for i in range(len(q_a)):
        f.write(q_a[i][1])
        f.write('\n')

f = open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\q&a.txt', 'r', encoding='utf-8')
w = open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\all_words_7_24.txt', 'a', encoding='utf-8')
partten = re.compile('\w*', re.L)
while True:
    line = f.readline()
    if line == '':
        break
    line = re.sub(partten, '', line)
    seg_list = jieba.lcut(line.strip(), cut_all=False)
    list1 = []
    for word in seg_list:
        if word not in stop_words and word != '':
            list1.append(word)
    w.write(' '.join(list1))
    w.write('\n')
w.close()
f.close()
print('finish')
# partten = re.compile(r'\w*', re.L)
partten = re.compile('[\u4e00-\u9fa5]')
text = u'小白底图*[[[]234230-=0/,<.，。的跌停'
a = re.findall(partten, text)
print(a[0])
print(re.sub(partten, '', text))
print(re.sub('[^a-zA-Z0-9]', "", text))
