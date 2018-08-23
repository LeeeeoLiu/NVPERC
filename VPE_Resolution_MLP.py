# coding=utf-8

import argparse
import cPickle as pickle
from datetime import datetime
import random
from Sentence import Sentence
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from model import MLP, MLP_Res, MLP_1, MLP_2, MLP_3, MLP_4, MLP_5, MLP_6
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from VPE_Detection_SVM import get_valid_data
import numpy as np
import logging

BATCH_SIZE = 64
EPOCH = 100
LR = 0.001
WD = 0.001

AUX_TYPE_LIST = ['do', 'be', 'so', 'have', 'to', 'modal']

def dataset_2_loader(dataset, data_key=0):
    data = []
    data_tag = []
    aux_type = []
    for _sent in dataset:
        if data_key == 0:
            data.append(_sent.input_vec_sum)
        elif data_key == 1:
            data.append(_sent.input_vec_sum_feature)
        elif data_key == 2:
            data.append(_sent.input_vec_hidden)
        elif data_key == 3:
            data.append(_sent.input_vec_hidden_feature)
        elif data_key == 4:
            data.append(_sent.input_vec_attention)
        elif data_key == 5:
            data.append(_sent.input_vec_attention_feature)
        data_tag.append(_sent.antecedent_label)
        aux_type.append(AUX_TYPE_LIST.index(_sent.aux_type))

    dtensor = torch.from_numpy(np.array(data))
    ttensor = torch.from_numpy(np.array(data_tag))
    aux_tensor = torch.from_numpy(np.array(aux_type))
    dataset = Data.TensorDataset(dtensor, ttensor, aux_tensor)
    Data.TensorDataset()

    data_loader = Data.DataLoader(
                dataset=dataset,
                batch_size=BATCH_SIZE, 
                shuffle=True,         
                num_workers=0,       
            )
    return data_loader


def evaludate(_model, _data_loader):
    TP, FP, FN, TN = 0, 0, 0, 0
    auxs_dict = {}
    for datas, tags, auxs in _data_loader:
        if args.gpu > -1 and torch.cuda.is_available():
            inputs = Variable(datas.float()).cuda()
            labels = Variable(tags.long()).cuda()
        else:
            inputs = Variable(datas.float())
            labels = Variable(tags.long())
        outputs = _model(inputs)
        if args.gpu > -1 and torch.cuda.is_available():
            outputs = outputs.cpu()
            labels = labels.cpu()
        predict = np.argmax(outputs.data.numpy(), 1)
        # predict = outputs.squeeze().data.numpy()
        truth = labels.data.numpy()
        for idx, p in enumerate(predict):
            _aux = AUX_TYPE_LIST[auxs[idx]]
            if not auxs_dict.has_key(_aux):
                auxs_dict[_aux] = [0, 0, 0, 0]
            aux_result = auxs_dict[_aux]
            if truth[idx] == 1 and predict[idx] == 1:
                TP += 1
                aux_result[0] += 1
            elif truth[idx] == 1 and predict[idx] == 0:
                FN += 1
                aux_result[1] += 1
            elif truth[idx] == 0 and predict[idx] == 1:
                FP += 1
                aux_result[2] += 1
            else:
                TN += 1
                aux_result[3] += 1
            auxs_dict[_aux] = aux_result

    if TP != 0:
        P = float(TP)/(TP + FP)
        R = float(TP)/(TP + FN)
        F1 = (2*P*R)/(P + R)
        Acc = float(TP+TN)/(TP+TN+FP+FN)
    else:
        P, R, F1, Acc = 0, 0, 0, 0
    all_results = {'all': [P, R, F1, Acc]}
    for _key in auxs_dict.keys():
        [_tp, _fn, _fp, _tn] = auxs_dict[_key]
        if _tp != 0:
            _p = float(_tp)/(_tp + _fp)
            _r = float(_tp)/(_tp + _fn)
            _f1 = (2*_p*_r)/(_p + _r)
            _acc = float(_tp+_tn)/(_tp+_tn+_fp+_fn)
        else:
            _p, _r, _f1, _acc = 0, 0, 0, 0
        all_results[_key] = [_p, _r, _f1, _acc]
    return all_results


if __name__ == '__main__':
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_test', type=bool, default=False, help="Type:bool. Whether use train-test proposed by Bos for training. \nDefalut:False")
    parser.add_argument('-aux', type=bool, default=False, help="Type:bool. Whether show classification report for each Auxiliary. \nDefalut:False")
    parser.add_argument('-gpu', type=int, default=-1, help="Use GPU Id \nDefalut:-1")
    parser.add_argument('-model', type=int, default=0, help="Choose Model \nDefalut:0(SUM)")
    

    args = parser.parse_args()
    ROOT_DATA_PATH = './datas/data_manager/'
    # parameters of logging
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='logs/VPE_Resolution_{}.log'.format(args.model), level=logging.DEBUG, format=LOG_FORMAT)
    with open('{}self_antecedent_generate_sentences.pkl'.format(ROOT_DATA_PATH),'r') as f:
        _sentences = pickle.load(f)
        if args.model == 0:
            model = MLP_1(400)
        elif args.model == 1:
            model = MLP_2(468)
        elif args.model == 2:
            model = MLP_3(6100)
        elif args.model == 3:
            model = MLP_4(6168)
        elif args.model == 4:
            model = MLP_5(1600)
        elif args.model == 5:
            model = MLP_6(1668)
        
        logging.info('=================') 
        if args.gpu > -1 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            model = model.cuda()
        
        # use 5-cross validation
        pos_data = [_sent for _sent in  _sentences if int(_sent.antecedent_label) == 1]
        neg_data = [_sent for _sent in  _sentences if int(_sent.antecedent_label) == 0]
        assert len(pos_data) != 0
        _pos_cross_valid_all = get_valid_data(pos_data)
        _neg_cross_valid_all = get_valid_data(neg_data)
        accuracy_dict = {}
        for i in range(5):
            [_pos_4_train, _pos_4_test] = _pos_cross_valid_all[i]
            [_neg_4_train, _neg_4_test] = _neg_cross_valid_all[i]
            train_data = _pos_4_train + random.sample(_neg_4_train, len(_pos_4_train)) 
            test_data = _pos_4_test + random.sample(_neg_4_test, len(_pos_4_test)) 
            logging.info('---------------------------------')
            logging.info('cross_validation id:' + str(i))
            logging.info('data for training:' + len(train_data).__str__())
            logging.info('data for testing:' + len(test_data).__str__())
            logging.info('---------------------------------')
            training_loader = dataset_2_loader(train_data, args.model)
            testing_loader = dataset_2_loader(test_data, args.model)
            opt = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=WD)
            loss_function = torch.nn.CrossEntropyLoss()
            for epoch in range(EPOCH):
                model.train()
                for step, (datas, tags, _) in enumerate(training_loader):
                    opt.zero_grad()
                    if args.gpu > -1 and torch.cuda.is_available():
                        inputs = Variable(datas.float()).cuda()
                        labels = Variable(tags.long()).cuda()
                    else:
                        inputs = Variable(datas.float())
                        labels = Variable(tags.long())
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    del outputs
                    del labels
                    logging.info('Epoch: {}/{}\tStep: {} \tLoss:{}'.format(epoch, EPOCH, step, loss.cpu().data))
                    loss.backward()
                    opt.step()
            model.eval()
            all_results = evaludate(model, testing_loader)
            logging.info('*********************')

            for _key in all_results.keys():
                _acc = all_results[_key][3]
                if _acc!=0:
                    try:
                        accuracy_dict[_key].append(_acc)
                    except KeyError:
                        accuracy_dict[_key]= [_acc]
        for _key in accuracy_dict.keys():
            aver_acc = np.array(accuracy_dict[_key]).mean()
            logging.info('{} average accuracy:{}'.format(_key, aver_acc))
    end_time = datetime.now()
    logging.info('Cost time:{}'.format(end_time-start_time))
