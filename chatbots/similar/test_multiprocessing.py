from multiprocessing import Pool
import os
from operator import itemgetter
import numpy as np
from gensim.models import word2vec
from jieba import posseg
import jieba

cos_similar = {}

model = word2vec.Word2Vec.load('D:\PyCharm\PyCharmProjects\simba\chatbots\similar\datebao_no_eng.model')
flag_w = {'x': 1.6, 'n': 1.2, 'nz': 1, 'vn': 1, 'uj': 0.5, 'v': 0.91, 'r': 0.5, 'm': 1, 'a': 1, 'c': 1, 'l': 1,
          'd': 0.8, 'y': 0.5}

'''
获得all_sep_questions_arr
'''


def get_all_sep_questions_arr():
    # flag_w = {'x':1.6,'n':1.5,'nz':1.6,'vn':1.6,'uj':0.01,'v':1.6,'r':0.4,'m':0.4,'a':0.5,'c':0.5,'l':0.6}
    # flag_w = {'x':2.5,'n':2.5,'nz':1.6,'vn':2.5,'uj':0.01,'v':2,'r':1,'m':1,'a':1,'c':1,'l':1}
    questions_list = []
    with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\sep_questions_7_24.txt', encoding='utf-8') as f:
        list2 = f.read().split('\n')
    for line in list2:
        line = line.split(' ')
        questions_list.append(line)
    all_sep_questions_arr = []
    for each_question in questions_list:
        line_arr = np.zeros((1, 200), dtype='float32')
        count = 0.0
        flag_sum = 0.0
        for word_f in each_question:
            word_f = word_f.split('_')
            word = word_f[0]
            flag = word_f[1]
            if word in model:
                count += 1

                if flag in flag_w.keys():
                    line_arr += (model[word] * flag_w[flag])
                    flag_sum += flag_w[flag]
                else:
                    line_arr += (model[word] * 1)
                    flag_sum += 1
        all_sep_questions_arr.append(line_arr / len(each_question))
        # all_sep_questions_arr.append(line_arr / flag_sum)
        # all_sep_questions_arr.append(line_arr/count)
    return questions_list, all_sep_questions_arr


'''
获得q_a
'''


def get_q_a():
    with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\q&a.txt', encoding='utf-8') as f:
        q_a_1 = f.read().split('\n')
    q = []
    a = []
    for k in range(len(q_a_1)):
        if k % 2 == 0:
            q.append(q_a_1[k])
        else:
            a.append(q_a_1[k])
    q_a = []
    for r in range(len(q)):
        q_a.append([q[r], a[r]])
    return q_a


'''
辅助函数计算余弦距离
'''


def vector_cosine(num, v1, v2):
    # print 'Run task %s (%s)...' % (num, os.getpid())
    result = num, 1 + v1.dot(v2.T) / (1 + np.sqrt(v1.dot(v1.T)) * np.sqrt(v2.dot(v2.T)))
    return result


'''
测试
'''
# def test(self, text, threshold):
#     seg_list = jieba.lcut(text.strip(), cut_all=False)
#     seg_words = []
#     for word in seg_list:
#         if word.encode('utf-8') not in stopwords and word != '':
#             seg_words.append(word)
#     score_dict = {}
#     for i in range(len(self.questions_list)):
#         each_score_lst = []
#         for input_word in seg_words:
#             similar_lst = []
#             if u'%s' % input_word in model:
#                 for word in self.questions_list[i]:
#                     word = word.split('_')[0]
#                     if u'%s' % word in model:
#                         similar_lst.append(model.similarity(u'%s' % word, u'%s' % input_word))
#                     else:
#                         similar_lst.append(0)
#             else:
#                 for word in self.questions_list[i]:
#                     similar_lst.append(0)
#             each_score_lst.append(similar_lst)
#
#         arr1 = np.array(each_score_lst)
#         # print arr1
#         # print '-----------'
#         similar_score = 0
#         count = 0.0
#         while np.max(arr1) > threshold:
#             similar_score += np.max(arr1)
#             count += 1
#             re = np.where(arr1 == np.max(arr1))
#             row = re[0][0]
#             column = re[1][0]
#             arr1[row, :] = 0
#             arr1[:, column] = 0
#         not_similar_arr = np.transpose(np.nonzero(arr1))
#         m = len(not_similar_arr)
#         if m != 0:
#             not_similar_score = 0.0
#             for j in range(m):
#                 not_similar_score += (1 - arr1[not_similar_arr[j][0], not_similar_arr[j][1]])
#                 # print not_similar_score
#             score_dict[str(j)] = similar_score / (similar_score + not_similar_score)
#         elif m == 0 and count != 0:
#             score_dict[str(j)] = similar_score / count
#         else:
#             score_dict[str(j)] = 0
#             # print j
#     sorted_score = sorted(score_dict.iteritems(), key=itemgetter(1), reverse=True)
#     print 'the input question is %s' % text
#     if sorted_score[0][1] > 0.8:
#         print 'the question is:%s the similarity is: %f' % (
#         self.q_a[int(sorted_score[0][0])][0], sorted_score[0][1])
#         print '------------------------------------------------------'
#         return self.q_a[int(sorted_score[0][0])][0]
#     else:
#         print 'the question is:%s the similarity is: %f' % (
#         self.q_a[int(sorted_score[0][0])][0], sorted_score[0][1])
#         print 'the question is:%s the similarity is: %f' % (
#         self.q_a[int(sorted_score[1][0])][0], sorted_score[1][1])
#         print 'the question is:%s the similarity is: %f' % (
#         self.q_a[int(sorted_score[2][0])][0], sorted_score[2][1])
#         print 'the question is:%s the similarity is: %f' % (
#         self.q_a[int(sorted_score[3][0])][0], sorted_score[3][1])
#         print 'the question is:%s the similarity is: %f' % (
#         self.q_a[int(sorted_score[4][0])][0], sorted_score[4][1])
#         print '------------------------------------------------------'
#         return [self.q_a[int(sorted_score[0][0])][0], self.q_a[int(sorted_score[1][0])][0],
#                 self.q_a[int(sorted_score[2][0])][0], self.q_a[int(sorted_score[3][0])][0],
#                 self.q_a[int(sorted_score[4][0])][0]]

'''
获得输入文本向量
'''


def get_text_vector(text_input):
    with open('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().split('\n')
    jieba.load_userdict('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\entitydict.txt')
    text_vector = np.zeros((1, 200), dtype='float32')
    seg_list = posseg.cut(text_input.strip())
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
    text_vector_result = text_vector / len(seg_words)
    return text_vector_result


'''
简单向量相加
'''


def vector_sum_similar(tuple1):
    num = tuple1[0]
    similar_score = tuple1[1]
    # text_vector = text_vector/count
    # text_vector = text_vector / flag_sum
    # print text_vector
    cos_similar[str(num)] = similar_score


'''
结果输出
'''


def output_answer(q_a):
    sorted_score = sorted(cos_similar.items(), key=itemgetter(1), reverse=True)
    if sorted_score[0][1] > 1.951:
        print('the question is:%s the similarity is: %f' % (
            q_a[int(sorted_score[0][0])][0], sorted_score[0][1]))
        print('------------------------------------------------------')
        return q_a[int(sorted_score[0][0])][0]
    else:
        print('the question is:%s the similarity is: %f' % (
            q_a[int(sorted_score[0][0])][0], sorted_score[0][1]))
        print('the question is:%s the similarity is: %f' % (
            q_a[int(sorted_score[1][0])][0], sorted_score[1][1]))
        print('the question is:%s the similarity is: %f' % (
            q_a[int(sorted_score[2][0])][0], sorted_score[2][1]))
        print('the question is:%s the similarity is: %f' % (
            q_a[int(sorted_score[3][0])][0], sorted_score[3][1]))
        print('the question is:%s the similarity is: %f' % (
            q_a[int(sorted_score[4][0])][0], sorted_score[4][1]))
        print('------------------------------------------------------')
        return [q_a[int(sorted_score[0][0])][0], q_a[int(sorted_score[1][0])][0],
                q_a[int(sorted_score[2][0])][0], q_a[int(sorted_score[3][0])][0],
                q_a[int(sorted_score[4][0])][0]]
        #     print 'the question is:%s the similarity is: %f' % (q_a[int(sorted_score[5][0])][0],sorted_score[5][1])
        #     print 'the question is:%s the similarity is: %f' % (q_a[int(sorted_score[6][0])][0],sorted_score[6][1])
        #     print 'the question is:%s the similarity is: %f' % (q_a[int(sorted_score[7][0])][0],sorted_score[7][1])
        # return cos_similar


def precise_recall(self, test_path):
    with open(test_path, 'r', encoding='utf-8') as f:
        test_text = f.read().split('\n')
    input_sentence = [line.split('\t', 1)[0].strip() for line in test_text]
    wished_questions = [line.split('\t')[1:] for line in test_text]
    recall = 0
    precise = 0
    for k in range(len(input_sentence)):
        count = 0
        similar_questions = self.vector_sum_similar(flag_w, model, input_sentence[k])
        # similar_questions = self.test(sentence,0.7)
        if isinstance(similar_questions, list):
            for each_question in similar_questions:
                if each_question in wished_questions[k]:
                    count += 1
            recall += (count / float(len(wished_questions[k])))
            precise += (count / float(len(similar_questions)))
        if isinstance(similar_questions, str):
            if similar_questions in wished_questions:
                recall += 1 / float(len(wished_questions[k]))
                precise += 1
    recall = recall / float(len(test_text))
    precise = precise / float(len(test_text))
    print('recall is %f' % recall)
    print('precise is %f' % precise)


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    text = '什么是海损'
    print('The input question is %s' % text)
    questions_list, all_sep_questions_arr = get_all_sep_questions_arr()
    text_vector = get_text_vector(text)
    p = Pool()
    for j in range(len(all_sep_questions_arr)):
        p.apply_async(vector_cosine, args=(j, text_vector, all_sep_questions_arr[j],), callback=vector_sum_similar)
    print('Waiting for all sub processes done...')
    p.close()
    p.join()
    print('All sub processes done.')
    output_answer(get_q_a())
