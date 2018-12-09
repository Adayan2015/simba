from operator import itemgetter
import numpy as np
from gensim.models import word2vec
from jieba import posseg
import jieba
import sys

sys.path.append("..")
with open('chatbots/similar/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().split('\n')
jieba.load_userdict('chatbots/similar/entitydict.txt')
model = word2vec.Word2Vec.load('chatbots/similar/datebao_no_eng.model')
flag_w = {'x': 1.6, 'n': 1.2, 'nz': 1, 'vn': 0.9, 'uj': 0.5, 'v': 0.8, 'r': 0.5, 'm': 0.6, 'a': 1, 'c': 0.9, 'l': 1,
          'd': 0.8, 'y': 0.5, 'nr': 1.2, 'i': 1.2}


class SimilarBot:

    def __init__(self):
        self.all_sep_questions_arr = np.load(r'chatbots/similar/all_sep_questions_arr.npy')
        self.all_sep_questions = self.get_all_sep_questions()
        self.q_a = self.get_q_a()

    '''
    预处理，将库中所有问题向量化，获得all_sep_questions_arr,并保存下来
    '''

    def get_all_sep_questions_arr(self):
        all_sep_questions = []
        with open('chatbots/similar/sep_questions_7_24.txt', 'r', encoding='utf-8') as ff:
            list2 = ff.read().strip().split('\n')
        for line in list2:
            line = line.split(' ')
            all_sep_questions.append(line)
        all_sep_questions_arr = []
        for each_question in all_sep_questions:
            line_arr = np.zeros((1, 200), dtype='float32')
            count = 0.0
            flag_sum = 0.0
            for word_f in each_question:
                word_f = word_f.split('_')
                word = word_f[0]
                flag = word_f[1]
                if u'%s' % word in model:
                    count += 1
                    if flag in flag_w.keys():
                        line_arr += (model[u'%s' % word] * flag_w[flag])
                        flag_sum += flag_w[flag]
                    else:
                        line_arr += (model[u'%s' % word] * 1)
                        flag_sum += 1
            all_sep_questions_arr.append(line_arr / len(each_question))
        np.save('all_sep_questions_arr.npy', all_sep_questions_arr)
        return all_sep_questions, all_sep_questions_arr

    def get_all_sep_questions(self):
        all_sep_questions = []
        with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\sep_questions_7_24.txt', 'r', encoding='utf-8') as ff:
            list2 = ff.read().strip().split('\n')
        for line in list2:
            line = line.split(' ')
            all_sep_questions.append(line)
        return all_sep_questions

    '''
    获得q_a,问题与答案集合，二维向量，[i][0]为第i个QA对的问题，[i][1]为答案
    '''

    def get_q_a(self):
        with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\q&a.txt', 'r', encoding='utf-8') as ff:
            q_a_1 = ff.read().split('\n')
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
        return q_a

    # 辅助函数计算余弦距离
    def vector_cosine(self, v1, v2):
        result = (v1.dot(v2.T)) / (np.math.exp(-15) + np.sqrt(v1.dot(v1.T)) * np.sqrt(v2.dot(v2.T)))
        return result

    # 方法二：语义jaccard（时间复杂度高）
    def test(self, text, threshold):
        seg_list = jieba.lcut(text.strip(), cut_all=False)
        seg_words = []
        for word in seg_list:
            if word.encode('utf-8') not in stopwords and word != '':
                seg_words.append(word)
        score_dict = {}
        for i in range(len(self.all_sep_questions)):
            each_score_lst = []
            for input_word in seg_words:
                similar_lst = []
                if input_word in model:
                    for word in self.all_sep_questions[i]:
                        word = word.split('_')[0]
                        if word in model:
                            similar_lst.append(model.similarity(word, input_word))
                        else:
                            similar_lst.append(0)
                else:
                    for word in self.all_sep_questions[i]:
                        similar_lst.append(0)
                each_score_lst.append(similar_lst)

            arr1 = np.array(each_score_lst)
            similar_score = 0
            count = 0.0
            while np.max(arr1) > threshold:
                similar_score += np.max(arr1)
                count += 1
                re = np.where(arr1 == np.max(arr1))
                row = re[0][0]
                column = re[1][0]
                arr1[row, :] = 0
                arr1[:, column] = 0
            not_similar_arr = np.transpose(np.nonzero(arr1))
            m = len(not_similar_arr)
            if m != 0:
                not_similar_score = 0.0
                for j in range(m):
                    not_similar_score += (1 - arr1[not_similar_arr[j][0], not_similar_arr[j][1]])

                score_dict[str(i)] = similar_score / (similar_score + not_similar_score)
            elif m == 0 and count != 0:
                score_dict[str(i)] = similar_score / count
            else:
                score_dict[str(i)] = 0

        sorted_score = sorted(score_dict.items(), key=itemgetter(1), reverse=True)

        print(sorted_score)
        print('the input question is %s' % text)
        if sorted_score[0][1] > 0.8:
            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[0][0])][0], sorted_score[0][1]))
            print('------------------------------------------------------')
            return self.q_a[int(sorted_score[0][0])][0]
        else:
            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[0][0])][0], sorted_score[0][1]))
            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[1][0])][0], sorted_score[1][1]))
            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[2][0])][0], sorted_score[2][1]))
            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[3][0])][0], sorted_score[3][1]))
            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[4][0])][0], sorted_score[4][1]))
            print('------------------------------------------------------')
            return [self.q_a[int(sorted_score[0][0])][0], self.q_a[int(sorted_score[1][0])][0],
                    self.q_a[int(sorted_score[2][0])][0], self.q_a[int(sorted_score[3][0])][0],
                    self.q_a[int(sorted_score[4][0])][0]]

    '''
    方法一：词向量乘权重相加
    '''

    def vector_sum_similar(self, text):
        text_vector = np.zeros((1, 200), dtype='float32')
        seg_list = posseg.cut(text.strip())
        seg_words = []
        for word, flag in seg_list:
            if word not in stopwords and word != '':
                seg_words.append(word + '_' + flag)
        count = 0.0
        flag_sum = 0.0
        for word_f in seg_words:
            word = word_f.split('_')[0]
            flag = word_f.split('_')[1]
            if word in model:
                count += 1
                if flag in flag_w.keys():
                    text_vector += (model[word] * flag_w[flag])
                    flag_sum += flag_w[flag]
                else:
                    text_vector += (model[word] * 1)
                    flag_sum += 1
        text_vector = text_vector / len(seg_words)

        cos_similar = {}
        for i in range(len(self.all_sep_questions_arr)):
            cos_similar[str(i)] = self.vector_cosine(text_vector, self.all_sep_questions_arr[i])
        sorted_score = sorted(cos_similar.items(), key=itemgetter(1), reverse=True)
        print('the input question is %s' % text)

        if sorted_score[0][1] > 0.9563:
            print("000000000000000000000000000000")
            print(sorted_score[0][0])
            print(sorted_score[0][1])
            print(self.q_a)
            print(
                'the question is:%s the similarity is: %f' % (
                    self.q_a[int(sorted_score[0][0]) - 1][0], sorted_score[0][1]))

            print('------------------------------------------------------')
        else:
            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[0][0])][0], sorted_score[0][1]))

            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[1][0])][0], sorted_score[1][1]))

            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[2][0])][0], sorted_score[2][1]))

            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[3][0])][0], sorted_score[3][1]))

            print(
                'the question is:%s the similarity is: %f' % (self.q_a[int(sorted_score[4][0])][0], sorted_score[4][1]))

            print('------------------------------------------------------')

        return sorted_score

    '''
    输入
    text：指定查询的句子
    num：要返回的匹配个数
    输出
    similar_questions
    similar_answers
    similar_score
    '''

    def get_similar_questions(self, text, num=3):
        similar_questions = []
        similar_answers = []
        similar_score = []
        sorted_score = self.vector_sum_similar(text)
        if num > 1:
            for i in range(num):
                similar_questions.append(u'%s' % self.q_a[int(sorted_score[i][0])][0])
                similar_answers.append(u'%s' % self.q_a[int(sorted_score[i][0])][1])
                similar_score.append('%f' % sorted_score[i][1])
            return similar_questions, similar_answers, similar_score
        elif num == 1:
            return self.q_a[int(sorted_score[0][0])][0], self.q_a[int(sorted_score[0][0])][1], '%f' % sorted_score[0][1]
        else:
            print("num can't be smaller than 1")

            return

    def output(self, text, num=3):
        output_questions, output_answers, output_scores = self.get_similar_questions(text, num)
        if isinstance(output_questions, list):
            if float(output_scores[0]) > 0.9563:
                return output_questions[0], output_answers[0], output_scores[0]
            else:
                return output_questions[:num], output_answers[:num], output_scores[:num]
        if isinstance(output_questions, str):
            return output_questions, output_answers, output_scores

    def precise_recall(self, test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_text = f.read().split('\n')
        input_sentence = [line.split('\t', 1)[0].strip() for line in test_text]
        wished_questions = [line.split('\t')[1:] for line in test_text]
        recall = 0
        top_1_recall = 0
        top_5_flag = 0
        top_1_flag = 0
        for i in range(len(input_sentence)):
            count = 0
            similar_questions_1, similar_answers_1, similar_score_1 = self.get_similar_questions(input_sentence[i], 1)
            similar_questions_5, similar_answers_5, similar_score_5 = self.get_similar_questions(input_sentence[i], 5)
            if isinstance(similar_questions_1, str):
                if similar_questions_1 in wished_questions[i]:
                    top_1_flag += 1
                    top_1_recall += 1 / float(len(wished_questions[i]))
            if isinstance(similar_questions_5, list):
                for each_question in similar_questions_5:
                    if each_question in wished_questions[i]:
                        count += 1
                recall += (count / float(len(wished_questions[i])))
            if count != 0:
                top_5_flag += 1
        recall = recall / float(len(test_text))
        top_1_precise = top_1_flag / float(len(input_sentence))
        top_5_precise = top_5_flag / float(len(input_sentence))
        print('top1 recall is %f' % top_1_recall)
        print('top1 precise is %f' % top_1_precise)
        print('top5 recall is %f' % recall)
        print('top5 precise is %f' % top_5_precise)


if __name__ == '__main__':
    chatbot = SimilarBot()
    # text = '我的车在外地出事了'
    # text = '什么是趸交'
    # sorted_score = chatbot.vector_sum_similar(text)

    # questions,answers,score=chatbot.output(text)
    # print questions
    # print answers

    chatbot.precise_recall('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\test_questions.txt')
    # all_sep_questions, all_sep_questions_arr = chatbot.get_all_sep_questions_arr()
    # chatbot.vector_cosine(all_sep_questions_arr[0],all_sep_questions_arr[0])
    # len(all_sep_questions)
    # Precise_Recall(r'F:\simi
    # lar\test_questions.txt')
    #
    # input_sentence,wished_questions = Precise_Recall(r'F:\similar\test_questions.txt')
    # if '医疗保险可以退保吗' in wished_questions[1]:
    #     print 'p'
    #
    # test_list = [[0,0],[0,1],[0,4]]
    # test_list = 'asdfa  dfa  d sdf 我是'
    # if type(test_list) is types.StringType:
    #     print 'o'
    # arr1 = np.array(test_list)
    # re = np.where(arr1 == np.max(arr1))
    # row = re[0][0]
    # column = re[1][0]
    # arr1[row,:] = 0
    # arr1[:,column] = 0
    # arr1[0,0]
    # arr1[1,0]
    # np.count_nonzero(arr1)
    # a = np.transpose(np.nonzero(arr1))
    # for i in range(len(a)):
    #     print arr1[a[i][0], a[i][1]]

    # np.count_nonzero(arr1)
    #
    # dict1 = {'1':265,'2':112,'3':323}
    # sorted_score = sorted(dict1.iteritems(),key=lambda d:d[1],reverse=True)
    #
    #
    #
    # a = (model[u'什么'] + model[u'是'] + model[u'意外险'] )/3
    # b = (model[u'意外险']+ model[u'是'] + model[u'什么'])/3
    # print vector_cosine(a,b)
    #
    # with open('questions.txt','a')as f:
    #     for i in range(len(q_a)):
    #         f.write(q_a[i][0])
    #         f.write('\n')
    #
    # seg_list = posseg.cut('如何给公司员工买意外险')
    # for word,flag in seg_list:
    #     print word,flag
    # seg_list = posseg.cut('什么是海损')
    # for word,flag in seg_list:
    #     print word,flag
    #
    # y1 = model.similarity(u"叫",u"是")
    # print u"woman和man的相似度为：", y1
    # print "--------\n"
    #

# with open('test_questions.txt', 'r') as f:
#     test_text = f.read().split('\n')
# input_sentence = [line.split('\t', 1)[0].strip() for line in test_text]
# wished_questions = [line.split('\t')[1:] for line in test_text]
# print len(wished_questions[1])
