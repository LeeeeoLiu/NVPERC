# coding=utf-8
import copy
import cPickle as pickle
from datetime import datetime
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_trigger_model(train_data):
    training_vec_feature = [_data.sen_vec_feature for _data in train_data]
    training_label = [_data.trigger_label for _data in train_data]
    clf_feature = SVC(C=100, gamma=0.5, kernel='rbf').fit(
        training_vec_feature, training_label)
    
    testing_vec_feature = [_data.sen_vec_feature for _data in test_data]
    testing_label = [_data.trigger_label for _data in test_data]
    y_true, y_pred = testing_label, clf_feature.predict(testing_vec_feature)
    _new_test_data = []
    for sen, _predic_lable in zip(test_data, y_pred):
        _sen = copy.deepcopy(sen)
        _sen.gold_trigger_label = sen.trigger_label
        _sen.trigger_label = _predic_lable
        _new_test_data.append(_sen)
    
    return clf_feature, _new_test_data

if __name__ == '__main__':
    start_time = datetime.now()
    ROOT_DATA_PATH = './datas/data_manager/'
    
    train_files = ['wsj.section{}'.format(i.__str__() if i > 9 else '0' + i.__str__()) for i in range(20)]
    test_files = ['wsj.section{}'.format(i.__str__() if i > 9 else '0' + i.__str__()) for i in range(20, 25)]

    with open('{}self_trigger_generate_sentences.pkl'.format(ROOT_DATA_PATH), 'r') as f:
        _sentences = pickle.load(f)
        # use train-test
        train_data = [_sent for _sent in _sentences if _sent.wsj_section in train_files]
        test_data = [_sent for _sent in _sentences if _sent.wsj_section in test_files]
        random.shuffle(train_data)
        random.shuffle(test_data)
        print '---------------------------------'
        print 'data for training:' + len(train_data).__str__()
        print 'data for testing:' + len(test_data).__str__()
        print '---------------------------------'
        trigger_model, new_test_data = train_trigger_model(train_data)

        with open('end_to_end/trigger_model.pkl', 'w') as f:
            pickle.dump(trigger_model, f, protocol=-1)

        with open('{}ete_trigger_generate_sentences.pkl'.format(ROOT_DATA_PATH), 'w') as f:
            pickle.dump(new_test_data, f, protocol=-1)

    end_time = datetime.now()
    print 'time: {}'.format(end_time- start_time)

