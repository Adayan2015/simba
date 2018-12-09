import jieba
import jieba.posseg as pseg
from jieba import analyse
import re
import os

productdict_path = os.path.join(os.getcwd(), "D:\\PyCharm\\PyCharmProjects\\simba\\resources\\productdict.txt")
jieba.load_userdict(productdict_path)

professiondict_path = os.path.join(os.getcwd(), "D:\\PyCharm\\PyCharmProjects\\simba\\resources\\professiondict.txt")
jieba.load_userdict(professiondict_path)

stopwords_path = os.path.join(os.getcwd(), "D:\\PyCharm\\PyCharmProjects\\simba\\resources\\stopwords.txt")
jieba.analyse.set_stop_words(stopwords_path)
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = f.readlines()

'''
@return [('word', 'flag'), ..]
'''


def cut(text):
    mixwords = []

    # 将语句进行分词操作，并标注词性
    for word, flag in pseg.cut(text):
        if word not in stopwords:
            mixwords.append((word, flag))
    return mixwords


def jieba_key_word(text, k):
    words = jieba.analyse.extract_tags(text, topK=k, withWeight=False, allowPOS=())
    for i in range(len(words)):
        if words[i] in stopwords:
            words.pop(i)
    return words


def keywords_dict(k):
    with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\sep_questions_7_24.txt', 'r', encoding='utf-8') as f:
        corpus = f.readlines()
    pattern = re.compile('\w+')
    for i in range(len(corpus)):
        corpus[i] = re.sub(pattern, '', corpus[i])
    text = '\n'.join(corpus)
    keywords = jieba_key_word(text, k)

    with open('D:\\PyCharm\\PyCharmProjects\\simba\\distinguish_words.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(keywords))


def jieba_analyses(text):
    keywords = analyse.extract_tags(text, topK=30)
    return keywords


if __name__ == '__main__':
    strs = ['全家桶保些什么?', '防爆警察是什么职业']
    for s in strs:
        print('Q:', s)
        for word, flag in cut(s):
            print(word, flag)

    # text = '晶算师中，购买保险后，奖励何时会发放到我的账户中'
    #
    # a = np.array([[1, 2, 3], [4, 5, 6]])
    # print(a.sum(axis=0))
    # print(a.sum(axis=1))
    # keywords_dict(30)
    # with open('D:\\PyCharm\\PyCharmProjects\\simba\\distinguish_words.txt', 'r', encoding='utf-8') as f:
    #     keywords = f.readlines()
    # for i in range(len(keywords)):
    #     keywords[i] = '%s' % keywords[i]
