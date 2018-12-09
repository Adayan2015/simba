from . import connect_mongo as mongodb_input
from . import data_preprocessing as preprocessing_2
from . import data_analyze as machine_learning
from . import output as mongodb_output


class Audit(object):
    # Data backup
    mongodb_input.mongo_input()

    """docstring for DialogBot."""

    def __init__(self):
        super(Audit, self).__init__()

    """
    @param content, {'type':, 'text':}
    @return True | None
    """

    def request(self, session_id, content):
        print('using Audit')

        if 'type' not in content:
            return session_id, None

        if content['type'] != 'json':
            return session_id, None
        if 'json' not in content:
            return session_id, None

        # Data preprocessing
        train_x_2, train_y_2, unlabeled_set_2 = preprocessing_2.create_feature_sets_and_labels('D:\\PyCharm\\PyCharmProjects\\simba\\data.json')

        # Data mining
        prediction = machine_learning.method(train_x_2, train_y_2, unlabeled_set_2)
        # if status == 4, 机器通过
        # if status == 5, 机器拒绝

        # Data write-in
        mongodb_output.mongo_update(prediction)

        return session_id, prediction