import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import jieba.posseg as pseg
from gensim.models import word2vec
from keras.models import Sequential
import jieba
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.noise import GaussianNoise
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing import sequence
from keras.optimizers import *
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding


def get_x_y():
    frame = pd.read_csv('D:\\PyCharm\\PyCharmProjects\\simba\\chatbots\\similar\\data_shuffled2.csv')
    y = frame['Trigger'][:1600]
    y = y.values
    x = np.load('zbx_questions_arr.npy')
    return np.mat(x), y


def randomforest(x, y):  # 110:78.1875% 60:77.9375%
    kfolds = KFold(len(x), 3)
    KFold(len(x), )
    thresholds = [i for i in range(50, 110, 10)]
    best_n = 0
    best_result = 0
    for n in thresholds:
        print('use%d' % n)

        results = []
        for train_idx, test_idx in enumerate(kfolds, start=1):
            train_x = x[train_idx]
            test_x = x[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]
            rfmod = RandomForestClassifier(n_estimators=n, max_features='auto', )
            rfmod.fit(train_x, train_y)
            acc = accuracy_score(test_y, rfmod.predict(test_x))
            results.append(acc)
        each_result = np.sum(results) / float(len(results))
        print(each_result)

        if each_result > best_result:
            best_result = each_result
            best_n = n
    print('best n is %d,best result is %f' % (best_n, best_result))


def svm_C(x, y):  # C=4 :78.875%
    kfolds = KFold(len(x), 3)
    # C_list = [i for i in range(3, 5, 1)]
    C_list = [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5]
    best_C = 0
    best_result = 0
    for each_C in C_list:
        print('use %f as C' % each_C)

        results = []
        for train_idx, test_idx in enumerate(kfolds, start=1):
            # 使用crossvalidation切分训练集和测试集
            train_x = x[train_idx]
            test_x = x[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]
            svmmod = SVC(kernel='rbf', C=each_C)
            svmmod.fit(train_x, train_y)
            acc = accuracy_score(test_y, svmmod.predict(test_x))
            results.append(acc)
        each_result = np.sum(results) / float(len(results))
        print(each_result)
        if each_result > best_result:
            best_result = each_result
            best_C = each_C
    print('best C is %d,best result is %f' % (best_C, best_result))


def svm(x, y):
    kfolds = KFold(len(x), 3)
    results = []
    for train_idx, test_idx in enumerate(kfolds, start=1):
        # 使用crossvalidation切分训练集和测试集
        train_x = x[train_idx]
        test_x = x[test_idx]
        train_y = y[train_idx]
        test_y = y[test_idx]
        svm = SVC(kernel='rbf', C=4)
        svm.fit(train_x, train_y)
        acc = accuracy_score(test_y, svm.predict(test_x))
        results.append(acc)
    each_result = np.sum(results) / float(len(results))
    print('the result is %f') % each_result


# feature_array = np.load(r'F:\Sogou_Project\new_information\MLP\sex_100_7.npy')
# user_df2 = pd.read_csv(r'F:\Sogou_Project\new_information\train_new_all_information.csv', na_values='0')
# age_df2 = user_df2[['content','sex']]
# age_df2 = age_df2.dropna()
# label_array= age_df2[['sex']].values
# for i in range(10):
#     print label_array[i][0]
# label_array[19574][0]
def keras_train_model(batch_size, layer, nb_epoch, x, y):
    kfolds = KFold(len(x), 3)
    nb_classes = 2
    # feature_array = np.load(r'F:\similar\cla_data\train_X.npy')
    # label_array = np.load(r'F:\similar\cla_data\train_Y.npy')
    # label_array_test = np.load(r'F:\similar\cla_data\test_Y.npy')
    # user_df2 = pd.read_csv(r'F:\Sogou_Project\new_information\train_all_info_only_n.csv', na_values='0')
    # age_df2 = user_df2[['content','gender']]
    # age_df2 = age_df2.dropna()
    # label_array= age_df2[['gender']].values
    a = []
    for i in range(len(y)):
        if int(y[i]) == 1:
            a.append([1, 0, 0])
        if int(y[i]) == 2:
            a.append([0, 1, 0])
        if int(y[i]) == 3:
            a.append([0, 0, 1])

    y = np.array(a)
    print(len(x), len(y))
    # label_array = np_utils.to_categorical(label_array, nb_classes)
    # print label_array

    # print len(label_array)
    # feature_length = len(feature_array[0])
    # print feature_length
    # x_train = feature_array[:int(0.8*len(feature_array))]
    # x_test = np.load(r'F:\similar\cla_data\test_X.npy')
    # x_train = feature_array
    # y_train = label_array[:int(0.8*len(label_array))]
    # y_test = np.array(b)
    # y_train = label_array
    # x_data_length = len(x_train)
    # print x_data_length
    results = []
    for train_idx, test_idx in enumerate(kfolds, start=1):
        # 使用crossvalidation切分训练集和测试集
        train_x = x[train_idx]
        test_x = x[test_idx]
        train_y = y[train_idx]
        test_y = y[test_idx]
        model = Sequential()
        # 第一层输入层
        model.add(Dense(layer, input_shape=(x.shape[1],)))
        model.add(Activation('relu'))
        model.add(Dropout(0.54))
        # 第二层
        model.add(Dense(layer))
        model.add(Activation('relu'))
        model.add(Dropout(0.54))
        # 输出层
        model.add(Dense(3))
        model.add(Activation('softmax'))

        # #LSTM
        #     model = Sequential()
        #     model.add(Embedding(feature_length, 256, input_length=feature_length))
        #     model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
        #     model.add(Dropout(0.5))
        #     model.add(Dense(2))
        #     model.add(Activation('sigmoid'))
        model.summary()
        # model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
        # adadelta 目前最好
        # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
        # adagrad = Adagrad(lr=0.01,epsilon=1e-6)
        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        # model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,shuffle=True,validation_split=0.1)
        model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                  validation_data=(test_x, test_y))
        score = model.evaluate(test_x, test_y, verbose=0)
        results.append(score[1])
        # print('Test accuracy:', score[1])
    # model.save(r'F:\Sogou_Project\new_information\cut_for_search\keras_age_model.h5')
    # result = model.predict_classes(x_test,batch_size=batch_size)
    # np.save('F:/Sogou_Project/new_information/MLP/result_education',result)
    result = np.sum(results) / float(len(results))
    print('the average result is %f') % result


def ensemble(x, y):
    kfolds = KFold(len(x), 3)
    results = []
    for train_idx, test_idx in enumerate(kfolds, start=1):
        # 使用crossvalidation切分训练集和测试集
        train_x = x[train_idx]
        test_x = x[test_idx]
        train_y = y[train_idx]
        test_y = y[test_idx]
        cf1 = KNeighborsClassifier(n_neighbors=16)
        cf2 = SVC(C=4)
        cf3 = RandomForestClassifier(n_estimators=110)
        cf4 = DecisionTreeClassifier(min_samples_split=295)
        vote = VotingClassifier(estimators=[('KNN', cf1), ('SVC', cf2), ('RF', cf3), ('DT', cf4)])
        # bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=16),max_samples=0.5,max_features=0.5,n_estimators=20)
        # bagging = BaggingClassifier(DecisionTreeClassifier(min_samples_split=295), max_samples=0.5, max_features=0.5)
        # adaboost = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=295),n_estimators=110)
        vote.fit(train_x, train_y)
        acc = accuracy_score(test_y, vote.predict(test_x))
        results.append(acc)
    each_result = np.sum(results) / float(len(results))
    print('the result is %f') % each_result


def test_decisiontree(x, y):  # min_samples_split = 295
    kfolds = KFold(len(x), 3)
    best_c = 0
    best_result = 0
    list_c = [i for i in range(200, 301, 5)]
    for each_c in list_c:
        print('use %d' % each_c)
        results = []
        for train_idx, test_idx in enumerate(kfolds, start=1):
            # 使用crossvalidation切分训练集和测试集
            train_x = x[train_idx]
            test_x = x[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]
            dtmod = DecisionTreeClassifier(min_samples_split=each_c)
            dtmod.fit(train_x, train_y)
            acc = accuracy_score(test_y, dtmod.predict(test_x))
            results.append(acc)
        each_result = np.sum(results) / float(len(results))
        print('Decision Tree model acc is %s' % each_result)
        if each_result > best_result:
            best_result = each_result
            best_c = each_c
    print('best C is %d,best result is %f') % (best_c, best_result)


def test_KNN(x, y):  # n_neighbors=16
    kfolds = KFold(len(x), 3)
    best_c = 0
    best_result = 0
    list_c = [i for i in range(9, 21, 1)]
    for each_c in list_c:
        print('use %d' % each_c)

        results = []
        for train_idx, test_idx in enumerate(kfolds, start=1):
            # 使用crossvalidation切分训练集和测试集
            train_x = x[train_idx]
            test_x = x[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]
            dtmod = KNeighborsClassifier(n_neighbors=each_c)
            dtmod.fit(train_x, train_y)
            acc = accuracy_score(test_y, dtmod.predict(test_x))
            results.append(acc)
        each_result = np.sum(results) / float(len(results))
        print('Decision Tree model acc is %s') % each_result

        if each_result > best_result:
            best_result = each_result
            best_c = each_c
    print('best C is %d,best result is %f') % (best_c, best_result)


def LR(x, y):
    kfolds = KFold(len(x), 3)
    results = []
    for train_idx, test_idx in enumerate(kfolds, start=1):
        # 使用crossvalidation切分训练集和测试集
        train_x = x[train_idx]
        test_x = x[test_idx]
        train_y = y[train_idx]
        test_y = y[test_idx]
        lr = LogisticRegression(C=3.8421052631578947, penalty='l2', dual=False, solver='liblinear', multi_class='ovr')
        # bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=16),max_samples=0.5,max_features=0.5,n_estimators=20)
        # bagging = BaggingClassifier(DecisionTreeClassifier(min_samples_split=295), max_samples=0.5, max_features=0.5)
        # adaboost = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=295),n_estimators=110)
        lr.fit(train_x, train_y)
        acc = accuracy_score(test_y, lr.predict(test_x))
        results.append(acc)
    each_result = np.sum(results) / float(len(results))
    print('the result is %f') % each_result


if __name__ == '__main__':
    x, y = get_x_y()
    # name_lists = ['age', 'sex', 'education']
    # keras_train_model(7,300,4,x,y)
    # test_decisiontree(x,y)
    # svm(x,y)
    LR(x, y)
    # ensemble(x,y)
    # test_KNN(x,y)
