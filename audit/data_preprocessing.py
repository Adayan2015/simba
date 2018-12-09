import numpy as np
import random
import json
import jieba
from itertools import compress
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Segment sentences in database
"""


def create_lexicon(data):
    # Sentence segmentation
    s = data['question']
    seg_data = jieba.cut(s)
    seg_data = ' '.join(seg_data)
    seg_data = str(seg_data)

    return seg_data


"""
Create a one hot array for each label
"""


def classification(data, n_classes):
    label = np.zeros(n_classes)
    flag = data['status']
    # 1 = 待审核
    if flag == 1:
        label[0] += 1
        # print ('1')
    # 2 = 已通过
    if flag == 2 or flag == 5:
        label[1] += 1
        # print ('2')

    # 3 = 删除
    if flag == 3:
        label[2] += 1
        # print ('3')

    # 4 = 已拒绝
    if flag == 4 or flag == 6:
        label[3] += 1
        # print ('4')

    return np.ndarray.tolist(label)


"""
Perform Tf-idf on data and select features
"""


def tfidf(seg_data):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(seg_data)
    feature_name = vectorizer.get_feature_names()
    # print "Length of features: ", len(feature_name)
    temp = sparse.csr_matrix.sum(tfidf, axis=0)
    tfidf_score = temp[0]
    tfidf_valuable = tfidf_score > 0.6
    tfidf_valuable = tfidf_valuable.tolist()[0]
    output_lexicon = list(compress(feature_name, tfidf_valuable))

    return output_lexicon


"""
Filter words using stopwords.txt
"""


def stopwords(seg_data):
    stopwords_dict = []
    lexicon = []
    with open('stopwords.txt', 'r', encoding='utf-8') as words:
        for cur in words.read():
            stopwords_dict.append(cur)
    for cur_sentence in seg_data:
        for cur_word in cur_sentence.split(' '):
            if cur_word not in stopwords_dict:
                lexicon.append(cur_word)
    return list(set(lexicon))


"""
Create a one hot matrix
"""


def sample_handling(seg_data, lexicon):
    features = np.zeros((len(seg_data), len(lexicon), 1))
    for cur in range(len(seg_data)):
        for word in seg_data[cur].split(" "):
            if word in lexicon:
                index_value = lexicon.index(word)
                features[cur][index_value][0] += 1
    print('Created a one hot matrix for features.')
    return features


"""
Create a tuple of feature and labels
"""


def create_feature_sets_and_labels(data, n_classes=4):
    segmented_sentence = []
    mongo_id = []
    labels = []

    featureset = []
    unlabeled_set = []

    with open(data, 'r', encoding='utf-8') as data_file:
        faq = json.load(data_file)
        for cur in faq:
            s = create_lexicon(cur)
            segmented_sentence.append(s)
            flag = classification(cur, n_classes)
            labels.append(flag)
            mongo_id.append(cur['_id'])
    # print "Length of segmented_sentence: ", len(segmented_sentence)
    # print "Length of labels: ", len(labels)
    lexicon = stopwords(segmented_sentence)
    features = sample_handling(segmented_sentence, lexicon)
    # print "Shape of features: ", np.asarray(features).shape
    # print "Shape of labels: ", np.asarray(labels).shape
    print("Status distribution: ", np.sum(labels, axis=0))
    # [ 2049.  5201.    99.   147.]
    flag = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            unlabeled_set.append([mongo_id[i], features[i]])
        elif labels[i][2] == 1:
            flag += 1
        else:
            featureset.append([features[i], labels[i]])
    # shuffle the featureset
    random.shuffle(featureset)
    print('Dataset is shuffled!')
    featureset = np.array(featureset)
    # testing_size = int(test_size*len(featureset))
    # slice the dataset into training and testing
    train_x = list(featureset[:, 0])
    train_y = list(featureset[:, 1])
    # test_x = list(featureset[:,0][-testing_size:])
    # test_y = list(featureset[:,1][-testing_size:])
    # return train_x, train_y, unlabeled_set
    return train_x, train_y, test_x, test_y, unlabeled_set


"""
Activation function
"""

train_x, train_y, test_x, test_y, unlabeled_set = create_feature_sets_and_labels(
    data='D:\\PyCharm\\PyCharmProjects\\simba\\data.json')

"""
Dump file in pickle
"""
# with open('preprocessed_data.p', 'wb') as f:
#    pickle.dump([train_x,train_y,test_x,test_y, unlabeled_set],f)
#    print 'Data stored in preprocessed_data.p!'
