# coding=utf-8

import argparse
import cPickle as pickle
from datetime import datetime
import random
from Sentence import Sentence
from sklearn.svm import SVC
from sklearn.metrics import classification_report

Auxiliary_Do = ['do']
Auxiliary_Be = ['be']
Auxiliary_So = ['so', 'same', 'likewise', 'opposite']
Auxiliary_Have = ['have']
Auxiliary_To = ['to']
Auxiliary_Modal = ['will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'modal']



def get_valid_data(datas):
    """ Split data into 5 parts.

    For cross valid experiment.

    Args:
        datas: list, contains all the data.

    Returns:
        cross_valid_data: dict, contains each validation experiment data. For example: 
            cross_valid_data = {
                1:[train_data, test_data],
                2:[...,...],
                ...}
    """
    cross_valid_data = {}
    _split_datas = [[_d for _idx, _d in enumerate(datas) if _idx % 5 == j]for j in range(5)]
    for i in range(5):
        _idx = [0, 1, 2, 3, 4]
        _idx.remove(i)
        cross_valid_data[i] = [[_data for _re_idx in _idx for _data in _split_datas[_re_idx]], _split_datas[i]]

    return cross_valid_data

def filter_aux(data, lable, aux):
    lable_Be = []
    lable_Do = []
    lable_To = []
    lable_Have = []
    lable_Modal = []
    lable_So = []

    data_Be = []
    data_Do = []
    data_To = []
    data_Have = []
    data_Modal = []
    data_So = []
    # print aux
    for i in range(len(aux)):
        if aux[i] in Auxiliary_Be:
            lable_Be.append(lable[i])
            data_Be.append(data[i])
        elif aux[i] in Auxiliary_Do:
            lable_Do.append(lable[i])
            data_Do.append(data[i])
        elif aux[i] in Auxiliary_To:
            lable_To.append(lable[i])
            data_To.append(data[i])
        elif aux[i] in Auxiliary_Have:
            lable_Have.append(lable[i])
            data_Have.append(data[i])
        elif aux[i] in Auxiliary_Modal:
            lable_Modal.append(lable[i])
            data_Modal.append(data[i])
        elif aux[i] in Auxiliary_So:
            lable_So.append(lable[i])
            data_So.append(data[i])
    # assert 1==0

    return [('be', data_Be, lable_Be), ('do', data_Do, lable_Do),
            ('to', data_To, lable_To), ('have', data_Have, lable_Have),
            ('modal', data_Modal, lable_Modal), ('so', data_So, lable_So)]


def train_and_test(train_data, test_data, TRAIN_TEST=False):
    training_vec = [_data.sen_vec for _data in train_data]
    training_vec_feature = [_data.sen_vec_feature for _data in train_data]
    training_label = [_data.trigger_label for _data in train_data]
    # training
    clf = SVC(C=100, gamma=0.5, kernel='rbf').fit(training_vec, training_label)
    clf_feature = SVC(C=100, gamma=0.5, kernel='rbf').fit(training_vec_feature, training_label)
    if TRAIN_TEST:
        with open('models/svm.pkl', 'w') as f:
            pickle.dump(clf, f, protocol=-1)
            print 'save model to models/' 
        with open('models/svm_f.pkl', 'w') as f:
            pickle.dump(clf_feature, f,protocol=-1)
    # testing
    testing_vec = [_data.sen_vec for _data in test_data]
    testing_vec_feature = [_data.sen_vec_feature for _data in test_data]
    testing_label = [_data.trigger_label for _data in test_data]
    testing_aux = [_data.aux_type for _data in test_data]
    if args.aux:
        # show each auxiliary report
        print '================================'
        print 'test svm'
        Auxs = filter_aux(testing_vec, testing_label, testing_aux)
        for (_aux_tag, _vec_test, _label_test) in Auxs:
            print 'Auxiliary:{}'.format(_aux_tag)
            if len(_vec_test) == 0:
                print '0   0   0'
            else:
                y_true, y_pred = _label_test, clf.predict(_vec_test)
                print(classification_report(y_true, y_pred, digits=4))
        print '================================'
        print 'test svm with feature'
        Auxs = filter_aux(testing_vec_feature, testing_label, testing_aux)
        for (_aux_tag, _vec_test, _label_test) in Auxs:
            print 'Auxiliary:{}'.format(_aux_tag)
            if len(_vec_test) == 0:
                print '0   0   0'
            else:
                y_true, y_pred = _label_test, clf_feature.predict(_vec_test)
                print(classification_report(y_true, y_pred, digits=4))
    else:
        # show overall report
        print 'svm:'        
        y_true, y_pred = testing_label, clf.predict(testing_vec)
        print(classification_report(y_true, y_pred, digits=4))
        analysis = ['{}\t{}\t{}\t{}'.format(y_t,y_p, sent.trigger, sent.sen)for y_t,y_p, sent in  zip(y_true, y_pred, test_data)]
        with open('error_analysis/VPE_Detection_SVM.txt', 'w') as f:
            f.write('\n'.join(analysis))
        print 'svm with feature:'
        y_true, y_pred = testing_label, clf_feature.predict(testing_vec_feature)
        print(classification_report(y_true, y_pred, digits=4))
        analysis = ['{}\t{}\t{}\t{}'.format(y_t,y_p, sent.trigger, sent.sen) for y_t,y_p, sent in  zip(y_true, y_pred, test_data)]
        with open('error_analysis/VPE_Detection_SVM_Feature.txt', 'w') as f:
            f.write('\n'.join(analysis))


if __name__ == '__main__':
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_test', type=bool, default=False,
                        help="Type:bool. Whether use train-test proposed by Bos for training. \nDefalut:False")
    parser.add_argument('-aux', type=bool, default=False,
                        help="Type:bool. Whether show classification report for each Auxiliary. \nDefalut:False")
    args = parser.parse_args()
    ROOT_DATA_PATH = './datas/data_manager/'

    with open('{}self_trigger_generate_sentences.pkl'.format(ROOT_DATA_PATH),'r') as f:
        _sentences = pickle.load(f)
        if args.train_test:
            # use train-test
            train_files = ['wsj.section{}'.format(i.__str__() if i > 9 else '0' + i.__str__()) for i in range(20)]
            test_files = ['wsj.section{}'.format(i.__str__() if i > 9 else '0' + i.__str__()) for i in range(20, 25)]
            train_data = [_sent for _sent in  _sentences if _sent.wsj_section in train_files]
            test_data = [_sent for _sent in  _sentences if _sent.wsj_section in test_files]
            random.shuffle(train_data)
            random.shuffle(test_data)
            print '---------------------------------'
            print 'data for training:' + len(train_data).__str__()
            print 'data for testing:' + len(test_data).__str__()
            print '---------------------------------'
            train_and_test(train_data, test_data, True)
        else:
            # use 5-cross validation
            pos_data = [_sent for _sent in  _sentences if _sent.trigger_label == 1]
            neg_data = [_sent for _sent in  _sentences if _sent.trigger_label == 0]
            assert len(pos_data) != 0
            _pos_cross_valid_all = get_valid_data(pos_data)
            _neg_cross_valid_all = get_valid_data(neg_data)
            for i in range(5):
                [_pos_4_train, _pos_4_test] = _pos_cross_valid_all[i]
                [_neg_4_train, _neg_4_test] = _neg_cross_valid_all[i]
                train_data = _pos_4_train + _neg_4_train
                test_data = _pos_4_test + _neg_4_test
                random.shuffle(train_data)
                random.shuffle(test_data)
                print '---------------------------------'
                print 'cross_validation id:' + str(i)
                print 'pos for training:' + len(_pos_4_train).__str__()
                print 'neg for training:' + len(_neg_4_train).__str__()
                print 'pos for testing:' + len(_pos_4_test).__str__()
                print 'neg for testing:' + len(_neg_4_test).__str__()
                print '---------------------------------'
                train_and_test(train_data, test_data)

