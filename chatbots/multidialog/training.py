import csv
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import jieba
import jieba.analyse
import random
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from gensim.models import Word2Vec

jieba.load_userdict('insurance_type.txt')
jieba.analyse.set_stop_words('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\stopwords.txt')


def create_lexicon(data):
    # extract the key words based on the customized dictionary and word properties
    seg_data = jieba.analyse.extract_tags(data, allowPOS=('b', 'h', 'j', 'k', 'n', 'ni', 'nz', 'v', 'r'))
    seg_data = ' '.join(seg_data)
    seg_data = str(seg_data)
    return seg_data


# TODO: UPDATE REGULARLY
def feature_extraction(seg_data):
    dictionary = [cur.split(' ') for cur in seg_data]
    # min_count is a filter, the word that appears lower than the min_count will be filtered out
    model = Word2Vec(dictionary, min_count=1)
    model.save('Word2Vec')
    vector_form = []
    for one_sentence in dictionary:
        sentence_vector = [model.wv[each_word] for each_word in one_sentence]
        vector_form.append(sum(sentence_vector))
    # model.train(more_sentences)
    return vector_form


def display_topics(model, feature_names, no_top_words, tf):
    LatentDirichletAllocation(n_topics=4, max_iter=5, learning_method='online', learning_offset=50.,
                              random_state=0).fit(tf)
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def preprocessing():
    trigger = []
    insurance_type = []
    question = []
    preprocessed_data = []
    firstline = True
    with open(
            '/Users/davidgu/Google Drive/Notability/Computer Science/DaTeBao/develop/data/data_shuffled_with_labels.csv') as csvfile:
        filereader = csv.reader(csvfile)
        for one_line in filereader:
            if firstline:
                firstline = False
                continue
            else:
                if one_line[0] != '':
                    trigger.append(one_line[0])
                    insurance_type.append(one_line[1])
                    s = create_lexicon(one_line[4])
                    question.append(s)
                else:
                    continue

    word_vector = feature_extraction(question)
    for i in range(len(trigger)):
        preprocessed_data.append([word_vector[i], trigger[i]])
    random.shuffle(preprocessed_data)
    preprocessed_data = np.array(preprocessed_data)
    testing_size = int(0.2 * len(preprocessed_data))

    # slice the dataset into training and testing
    train_x = list(preprocessed_data[:, 0])
    train_y = list(preprocessed_data[:, 1])
    test_x = list(preprocessed_data[:, 0][-testing_size:])
    test_y = list(preprocessed_data[:, 1][-testing_size:])
    print('finished preprocessing')
    return train_x, train_y, test_x, test_y


def machine_learning(train_x, train_y, test_x, test_y):
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    total_X = []
    total_Y = []
    for each in train_x:
        train_X.append(each)
        total_X.append(each)
    for each in train_y:
        train_Y.append(each)
        total_Y.append(each)
    for each in test_x:
        test_X.append(each)
        total_X.append(each)
    for each in test_y:
        test_Y.append(each)
        total_Y.append(each)

    train_X = np.squeeze(train_X)
    test_X = np.squeeze(test_X)

    test_X = np.squeeze(total_X)

    clf1 = RandomForestClassifier(n_estimators=100)
    clf2 = LinearSVC()
    clf3 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), algorithm="SAMME", n_estimators=100,
                              learning_rate=1)

    model = VotingClassifier(estimators=[('rf', clf1), ('ada', clf2), ('knn', clf3)], voting='hard')

    model.fit(total_X, total_Y)
    joblib.dump(model, 'machine_learning_with_voting_clf.pkl')
    print('job done')

    """
    Cs = np.array(np.arange(100, 1000, 10))

    grid = GridSearchCV(estimator=model, param_grid=dict(n_estimators = Cs))
    grid.fit(train_X, train_Y)

    plt.plot(sorted(grid.cv_results_['mean_test_score']))
    plt.show()

    model.fit(train_X, train_Y)
    predicted_Y = model.predict(test_X)
    print model.score(test_X, test_Y)
    print confusion_matrix(test_Y, predicted_Y)

    for clf, label in zip([clf1, clf2, clf3, model], ['Logistic Regression', 'Random Forest', 'AdaBoost', 'Ensemble']):
        scores = cross_val_score(clf, train_X, train_Y, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    """


train_x, train_y, test_x, test_y = preprocessing()
machine_learning(train_x, train_y, test_x, test_y)
