# coding=utf-8
import copy
import cPickle as pickle
from datetime import datetime
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from model import MLP, MLP_Res, MLP_1, MLP_2, MLP_3, MLP_4, MLP_5, MLP_6
import logging
import argparse
import torch
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import os



BATCH_SIZE = 64
EPOCH = 100
LR = 0.001
WD = 0.001

def dataset_2_loader(dataset, data_key=0, _batch_size=BATCH_SIZE):
    data = []
    data_tag = []
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

    dtensor = torch.from_numpy(np.array(data))
    ttensor = torch.from_numpy(np.array(data_tag))
    dataset = Data.TensorDataset(dtensor, ttensor)
    Data.TensorDataset()

    data_loader = Data.DataLoader(
        dataset=dataset,
        batch_size=_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return data_loader


def train_model(_train_data, args, data_key=0):
    if data_key == 0:
        model = MLP_1(400)
    elif data_key == 1:
        model = MLP_2(468)
    elif data_key == 2:
        model = MLP_3(6100)
    elif data_key == 3:
        model = MLP_4(6168)
    elif data_key == 4:
        model = MLP_5(1600)
    elif data_key == 5:
        model = MLP_6(1668)

    logging.info('=================')
    if args.gpu > -1 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    training_loader = dataset_2_loader(_train_data, data_key)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        model.train()
        for step, (datas, tags) in enumerate(training_loader):
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
            logging.info(
                'Epoch: {}/{}\tStep: {} \tLoss:{}'.format(epoch, EPOCH, step, loss.cpu().data))
            loss.backward()
            opt.step()
    return model


def evaluate_and_predict(_model, _data_loader, _data_set, args):
    TP, FP, FN, TN = 0, 0, 0, 0
    analysis = ['GroundTruth_Tri_Label\t||Prediction_Tri_Label\t||Trigger\t||GroundTruth_Ant_Label\t||Prediction_Ant_Lable\t||Antecedent\t||Test_Instance']
    for idx, (datas, tags) in enumerate(_data_loader):
        _tmp_sentence = _data_set[idx]
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
        predict = np.argmax(outputs.data.numpy(), 1)[0]
        truth = labels.data.numpy()[0]
        if _tmp_sentence.gold_trigger_label == _tmp_sentence.trigger_label and _tmp_sentence.trigger_label == 1 and _tmp_sentence.antecedent_label == 1:
            truth == 1
        else:
            truth == 0

        _tmp_analysis = '{}\t||{}\t||{}\t||{}\t||{}\t||{}\t||{}'.format(
            _tmp_sentence.gold_trigger_label, _tmp_sentence.trigger_label, _tmp_sentence.trigger, truth, predict, _tmp_sentence.antecedent, _tmp_sentence.sen)
        analysis.append(_tmp_analysis)

        if truth == 1 and predict == 1:
            TP += 1
        elif truth == 1 and predict == 0:
            FN += 1
        elif truth == 0 and predict == 1:
            FP += 1
        else:
            TN += 1

    with open('end_to_end/VPE_Resolusion_Error_Analysis_{}.txt'.format(args.model), 'w') as f:
        f.write('\n'.join(analysis))

    if TP != 0:
        P = float(TP)/(TP + FP)
        R = float(TP)/(TP + FN)
        F1 = (2*P*R)/(P + R)
    else:
        P, R, F1 = 0, 0, 0
    return P, R, F1


if __name__ == '__main__':
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=-1,
                        help="Use GPU Id \nDefalut:-1")
    parser.add_argument('-model', type=int, default=0,
                        help="Choose Model \nDefalut:0(SUM)")

    args = parser.parse_args()
    ROOT_DATA_PATH = './datas/data_manager/'

    train_files = ['wsj.section{}'.format(
        i.__str__() if i > 9 else '0' + i.__str__()) for i in range(20)]
    test_files = ['wsj.section{}'.format(
        i.__str__() if i > 9 else '0' + i.__str__()) for i in range(20, 25)]

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='end_to_end/ETE_VPE_Resolution_{}.log'.format(
        args.model), level=logging.DEBUG, format=LOG_FORMAT)

    with open('{}self_antecedent_generate_sentences.pkl'.format(ROOT_DATA_PATH), 'r') as f:
        _sentences = pickle.load(f)
        train_data = [
            _sent for _sent in _sentences if _sent.wsj_section in train_files]
        pos_data = [_sent for _sent in train_data if int(
            _sent.antecedent_label) == 1]
        neg_data = [_sent for _sent in train_data if int(
            _sent.antecedent_label) == 0]
        train_data = pos_data + random.sample(neg_data, len(pos_data))
        random.shuffle(train_data)
        logging.info('---------------------------------')
        logging.info('data for training:' + len(train_data).__str__())
        logging.info('---------------------------------')
        model = train_model(train_data,args, args.model)

    torch.save(model.state_dict(), 'end_to_end/model_{}.pkl'.format(args.model))

    with open('{}ete_antecedent_generate_sentences.pkl'.format(ROOT_DATA_PATH), 'r') as f:
        test_data = pickle.load(f)
        random.shuffle(test_data)
        logging.info('start evaluating')
        testing_loader = dataset_2_loader(test_data, args.model, _batch_size=1)
        P, R, F1 = evaluate_and_predict(model, testing_loader, test_data, args)
        logging.info('P:{}\tR:{}\tF1:{}'.format(P, R, F1))

    end_time = datetime.now()
    logging.info('time: {}'.format(end_time - start_time))
