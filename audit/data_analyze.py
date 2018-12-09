import numpy as np
from sklearn.externals import joblib


def re_format_data(train_x, train_y, unlabeled):
    train_X = np.squeeze(train_x)
    train_Y = np.zeros(len(train_y))
    for i in range(len(train_y)):
        train_Y[i] = train_y[i].index(1) + 1

    data = []
    data_id = []
    for i in unlabeled:
        data.append(i[1])
        data_id.append(i[0])
    datas = np.squeeze(data)

    return train_X, train_Y, datas, data_id


def method(train_x, train_y, validation):
    train_X, train_Y, data, data_id = re_format_data(train_x, train_y, validation)
    # clf_LinearSVC = LinearSVC(C = 0.2, max_iter = 5)
    # scores = cross_val_score(clf_LinearSVC, train_X, train_Y, cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # clf_LinearSVC.fit(train_X, train_Y)
    result = []
    clf = joblib.load('D:\\PyCharm\\PyCharmProjects\\simba\\audit\\status_best_model.p')
    predicted = clf.predict(data)

    for i in range(len(predicted)):
        if predicted[i] == 2:
            result.append([data_id[i], 5])
        if predicted[i] == 4:
            result.append([data_id[i], 6])

    return result
