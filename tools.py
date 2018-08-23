# coding = utf-8
# -*- coding: utf-8 -*-
import os
import pickle
import re
import numpy as np
from datetime import datetime
from pattern.en import lemma
from nltk.tree import Tree
import random
import string
import nltk
from nltk import word_tokenize
import fasttext
from nltk.tree import ParentedTree
from scipy import spatial
import math

jar_path = "/users4/yxliu/Leeeeo_toolkit/berkeleyparser/BerkeleyParser-1.7.jar"
gr_path = "/users4/yxliu/Leeeeo_toolkit/berkeleyparser/eng_sm6.gr"

from BerkeleyParser import parser
berkeleyp = parser(jar_path, gr_path)
# tree = berkeleyp.parse("This is an apple")
# print tree

# 目前发现的trigger的规则
# ,/. so/So/or/nor/but/while [XXX] do/to/did/does
# as [XXX] were/do/does/did
# than  [XXX] do/is/had/has
# Verb 后马上就结束的
#   ,/. have [XXX] ,/.
#   [XXX] wasn’t/ would/ do/ might/have to [XXX] ,/.
# 从句 all the/the way/that/who/and [XXX] does/will/can
# the same [XXX] do
# doing/do [XXX] the same/so
# If it is/does/isn’t

start_time = datetime.now()
# 将正则表达式编译成Pattern对象

rule1 = re.compile(
    r'(?:[,.]\s|)(?:and\s|)(?:so|So|or|nor|but|while|)\s(?:[A-Za-z0-9_\'\.\-]*\s){0,2}(?:do|to|did|does|is|would|may)')
rule2 = re.compile(
    r'(?:[A-Za-z0-9_\'\.\-]*\s){0,2}(?:as)\s(?:[A-Za-z0-9_\'\.\-]*\s){0,3}(?:were|do|does|did|is|been|are|was|have|has|had|can|could)')
rule3 = re.compile(r'than\s(?:[A-Za-z0-9_\'\.\-]*\s){0,3}(?:do|is|had|has|did|was|were|are|could|would|can)')
rule4 = re.compile(
    r'(?:[A-Za-z0-9_\'\.\-]*\s){0,2}(?:would|do|might|have\sto|was|does|to|did)\s(?:[A-Za-z0-9_\'\.\-]*\s){0,2}[,.]')
rule5 = re.compile(
    r'(?:[,.]|)\s(?:have|\'s|be|ca|n\'t|is|was|were|are|would|can|should|Should|Can|Would|and|And|will|did|had)\s(?:[A-Za-z0-9_\'\.\-]*\s){0,3}(?:[,.?]|\'\')')
rule6 = re.compile(
    r'(?:all\sthe|the\sway|that|who|and|when|way)\s(?:[A-Za-z0-9_\'\.\-]*\s){0,2}(?:does|will|can|do|could)')
rule7 = re.compile(r'the\ssame\s(?:[A-Za-z0-9_\'\.\-]*\s){0,2}')
rule8 = re.compile(r'(?:doing|do|does|did)\s(?:[A-Za-z0-9_\'\.\-]*\s){0,2}(?:the\ssame|so)')
rule9 = re.compile(r'If it (?:is|does)')

rule = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
RULE = rule

match = []

pos_tag_idx = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
               'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
               'VBZ', 'WDT', 'WP', 'WP$', 'WRB', ',', '.', '$', '``', "''", ':', '(', ')']
verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
trigger_pos = ['VBZ', 'VBP', 'VBD', 'VB', 'MD', 'TO', 'VBN', 'VBG', 'RB']
trigger_list = ['is', 'do', 'were', 'did', 'does', 'had', 'have', 'would', 'to', 'was', "'s", 'has', 'are', 'been',
                'be', 'will', 'can', 'doing', 'ca', 'Should', 'might', 'should', "'re", 'may', 'done', 'could', 'wo',
                'must', "n't"]
words_before_trigger = ['so', 'So', ',', '.', 'as', 'nor',
                        'and', 'than', 'but', 'that', 'who', 'which', 'But']
words_after_trigger = ['n\'t', 'too', 'so', 'not',
                       'those', 'that', 'this', 'it', 'other', ',', '.']

COMMON_TRIGGER_POS_LIST = ['VBZ', 'VBP', 'VBD',
                           'VB', 'MD', 'TO', 'VBN', 'VBG', 'RB']
COMMON_TRIGGER_LIST = ['is', 'do', 'were', 'did', 'does', 'had', 'have', 'would', 'to', 'was', "'s", 'has', 'are', 'been',
                       'be', 'will', 'can', 'doing', 'ca', 'Should', 'might', 'should', "'re", 'may', 'done', 'could', 'wo',
                       'must', "n't"]

POS_2_INDEX = pos_tag_idx

TAG_2_INDEX = ['B', 'I', 'O', 'E', 'S', 'T']

POS_2_INDEX_LENGTH = len(POS_2_INDEX)

WORDS_BEFORE_TRIGGER = words_before_trigger
WORDS_AFTER_TRIGGER = words_after_trigger
AUXILIARY_TYPE = ['do', 'be', 'have', 'modal', 'to', 'so']

VERB_LIST = verb_list

add_feature_aux_word_2_idx = []
add_feature_aux_lemma_2_idx = []
add_feature_aux_type_2_idx = []

Auxiliary_Do = ['do']
Auxiliary_Be = ['be']
Auxiliary_So = ['so', 'same', 'likewise', 'opposite']
Auxiliary_Have = ['have']
Auxiliary_To = ['to']
Auxiliary_Modal = ['will', 'would', 'can', 'could', 'should', 'may', 'might', 'must']


# Aux_list = [Auxiliary_Do, Auxiliary_Be, Auxiliary_So, Auxiliary_Have, Auxiliary_To, Auxiliary_Modal]
Aux_list = Auxiliary_Do + Auxiliary_Be + Auxiliary_So + Auxiliary_Have + Auxiliary_To + Auxiliary_Modal
# Aux_list = [Auxiliary_So]


ft_model = fasttext.load_model('./datas/fasttext_model_4_all_wsj.bin')


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


def get_data_and_lable_trigger(data_list):
    """ Get data for training.
    
    Args:
        data_list: list, For example:
            [([1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0], 1, 'do', u'do'),...]
    Returns:
        data: list, contains training vecs.
        label: list, contains training labels.
        trigger: list, contains training trigger word.
        aux: list, contains training trigger type.
    """
    data = [_data[0] for _data in data_list]
    label = [_data[1] for _data in data_list]
    trigger = [_data[2] for _data in data_list]
    aux = [_data[3] for _data in data_list]
    return data, label, trigger, aux


def read_corpus(data_files, ADD_FEATURE=False):
    """ Read and process data from corpus.

    Args:
        data_files: list, contains the dataset files. For example:
            ['../datas/wsj_tagged/wsj.section00', ..., '../datas/data_4_vpe_neg_2']

    Returns:
        pos_data: list, contains the positive data for training. For example:
            (vec, data_lable[index], trigger[index], a_tpye)
        neg_data: list, contains the negative data for training.
    """
    pos_data = []
    neg_data = []
    for data_file in data_files:
        _file_path = 'datas/dataset_tagged/{}'.format(data_file)
        with open(_file_path, 'r')as f:
            print 'process file: {}'.format(_file_path)
            for line in f:
                line = line.replace('\xef\xbb\xbf', '').replace('\t', '').replace('\r', '')
                if line:
                    vec_list, data_lable, trigger, aux_type = sen2vec(line, f.name, ADD_FEATURE)
                    if len(vec_list) != len(data_lable):
                        print data_lable
                    for index, vec in enumerate(vec_list):
                        a_tpye = aux_type[index] if aux_type[index] in Aux_list else 'others'
                        if data_lable[index]:
                            pos_data.append((vec, data_lable[index], trigger[index], a_tpye))
                        else:
                            neg_data.append((vec, data_lable[index], trigger[index], a_tpye))
    return pos_data, neg_data


def prepare_data_cross_validation(pos_data, neg_data, ADD_FEATURE=False):
    """ Prepare data for training
    
    Args:
        pos_data: list, contains all the positive data.
        neg_data: list, contains all the positive data.
    
    Returns:
        None
    """

    _pos_cross_valid_all = get_valid_data(pos_data)
    _neg_cross_valid_all = get_valid_data(neg_data)

    for i in range(5):
        [_pos_4_train, _pos_4_test] = _pos_cross_valid_all[i]
        [_neg_4_train, _neg_4_test] = _neg_cross_valid_all[i]
        _neg_4_train = random.sample(_neg_4_train, len(_pos_4_train))
        test_data = _pos_4_test + _neg_4_test[:len(_pos_4_test)]
        train_data = _pos_4_train + _neg_4_train
        print '---------------------------------'
        print 'cross_validation id:' + str(i)
        print 'pos for training:' + len(_pos_4_train).__str__()
        print 'neg for training:' + len(_neg_4_train).__str__()
        print 'pos for testing:' + len(_pos_4_test).__str__()
        print 'neg for testing:' + len(_neg_4_test[:len(_pos_4_test)]).__str__()
        print '---------------------------------'
        vec_train, label_train, trigger_train, aux_train = get_data_and_lable_trigger(train_data)
        vec_test, label_test, trigger_test, aux_test = get_data_and_lable_trigger(test_data)
        datas = [(np.array(vec_train), np.array(label_train),
                  trigger_train, aux_train, np.array(vec_test),
                  np.array(label_test), trigger_test, aux_test)]
        _path_to_save_datas = '5-cross_data/prepared_datas_with_feature_{}'.format(i) if ADD_FEATURE else '5-cross_data/prepared_datas_{}'.format(i)
        with open(_path_to_save_datas, 'w') as f:
            pickle.dump(datas, f)



def prepare_data(train_pos_data, train_neg_data, test_pos_data, test_neg_data, ADD_FEATURE=False):
    """ Prepare data for training
    
    Args:
        train_pos_data: list, contains all the positive data.
        train_neg_data: list, contains all the negative data.
        test_pos_data: list, contains all the positive data.
        test_neg_data: list, contains all the negative data.
    
    Returns:
        None
    """
    train_neg_data = random.sample(train_neg_data, len(train_pos_data))
    train_data = train_pos_data + train_neg_data
    test_neg_data = random.sample(test_neg_data, len(test_pos_data))
    test_data = test_pos_data + test_neg_data
    print '---------------------------------'
    print 'pos for training:{}'.format(len(train_pos_data))
    print 'neg for training:{}'.format(len(train_neg_data))
    print 'pos for testing:{}'.format(len(test_pos_data))
    print 'neg for testing:{}'.format(len(test_neg_data))
    print '---------------------------------'
    dataset = {
        'train':train_data,
        'test':test_data
    }
    for key in dataset.keys():
        vec, label, trigger, aux = get_data_and_lable_trigger(dataset[key])
        datas = [(np.array(vec), np.array(label), trigger, aux)]
        _path_to_save_datas = 'train-test_data/prepared_datas_with_feature_{}'.format(key) if ADD_FEATURE else 'train-test_data/prepared_datas_{}'.format(key)
        with open(_path_to_save_datas, 'w') as f:
            pickle.dump(datas, f)



def sen2vec(input_sent, file_name, ADD_FEATURE=False):
    vec_4_training_list = []
    trigger = []
    data_label = []
    aux_type = []
    words = input_sent.split()
    sen = ''
    for index, word in enumerate(words):
        # if word:
        # 7\/8/CD/O
        if '\\' in word:
            # print file_name
            # print word
            w = word.split('/')[0] + word.split('/')[1]
            pos = word.split('/')[2]
            tag = word.split('/')[3]
        else:
            w = word.split('/')[0]
            pos = word.split('/')[1]
            tag = word.split('/')[2]
        sen += w
        sen += ' '

    for index, word in enumerate(words):
        # if word:
        # 7\/8/CD/O
        if '\\' in word:
            # print file_name
            # print word
            w = word.split('/')[0] + word.split('/')[1]
            pos = word.split('/')[2]
            tag = word.split('/')[3]
        else:
            w = word.split('/')[0]
            pos = word.split('/')[1]
            tag = word.split('/')[2]
        # sen += w
        # sen += ' '
        if pos in trigger_pos:
            vec_4_training = [1 if w in trigger_list else 0]
            if index == 0:
                vec_4_training.append(-1)
                vec_4_training.append(-1)
                vec_4_training.append(-1)
            elif index == 1:
                vec_4_training.append(-1)
                vec_4_training.append(-1)
                vec_4_training.append(pos_tag_idx.index(
                    words[index - 1].split('/')[1] if '\\' not in words[index - 1] else words[index - 1].split('/')[
                        2]) / len(pos_tag_idx))
            elif index == 2:
                vec_4_training.append(-1)
                vec_4_training.append(pos_tag_idx.index(
                    words[index - 2].split('/')[1] if '\\' not in words[index - 2] else words[index - 2].split('/')[
                        2]) / len(pos_tag_idx))
                vec_4_training.append(pos_tag_idx.index(
                    words[index - 1].split('/')[1] if '\\' not in words[index - 1] else words[index - 1].split('/')[
                        2]) / len(pos_tag_idx))
            else:
                vec_4_training.append(pos_tag_idx.index(
                    words[index - 3].split('/')[1] if '\\' not in words[index - 3] else words[index - 3].split('/')[
                        2]) / len(pos_tag_idx))
                vec_4_training.append(pos_tag_idx.index(
                    words[index - 2].split('/')[1] if '\\' not in words[index - 2] else words[index - 2].split('/')[
                        2]) / len(pos_tag_idx))
                vec_4_training.append(pos_tag_idx.index(
                    words[index - 1].split('/')[1] if '\\' not in words[index - 1] else words[index - 1].split('/')[
                        2]) / len(pos_tag_idx))
            vec_4_training.append(pos_tag_idx.index(pos) / len(pos_tag_idx))
            word_size = len(words)
            if index == word_size - 3:
                vec_4_training.append(pos_tag_idx.index(
                    words[index + 1].split('/')[1] if '\\' not in words[index + 1] else words[index + 1].split('/')[
                        2]) / len(pos_tag_idx))
                vec_4_training.append(pos_tag_idx.index(
                    words[index + 2].split('/')[1] if '\\' not in words[index + 2] else words[index + 2].split('/')[
                        2]) / len(pos_tag_idx))
                vec_4_training.append(-1)
            elif index == word_size - 2:
                vec_4_training.append(pos_tag_idx.index(
                    words[index + 1].split('/')[1] if '\\' not in words[index + 1] else words[index + 1].split('/')[
                        2]) / len(pos_tag_idx))
                vec_4_training.append(-1)
                vec_4_training.append(-1)

            elif index == word_size - 1:
                vec_4_training.append(-1)
                vec_4_training.append(-1)
                vec_4_training.append(-1)
            else:
                vec_4_training.append(pos_tag_idx.index(
                    words[index + 1].split('/')[1] if '\\' not in words[index + 1] else words[index + 1].split('/')[
                        2]) / len(pos_tag_idx))
                # print file_name + ':' + input_sent
                vec_4_training.append(pos_tag_idx.index(
                    words[index + 2].split('/')[1] if '\\' not in words[index + 2] else words[index + 2].split('/')[
                        2]) / len(pos_tag_idx))
                vec_4_training.append(pos_tag_idx.index(
                    words[index + 3].split('/')[1] if '\\' not in words[index + 3] else words[index + 3].split('/')[
                        2]) / len(pos_tag_idx))

            tmp_sen = ['none'] + words + ['none']
            if '\\' in tmp_sen[index-1]:
                be_word = tmp_sen[index-1].split('/')[0] + tmp_sen[index-1].split('/')[1]
            else:
                be_word = tmp_sen[index-1].split('/')[0]
            if '\\' in tmp_sen[index + 1]:
                af_word = tmp_sen[index + 1].split('/')[0] + tmp_sen[index + 1].split('/')[1]
            else:
                af_word = tmp_sen[index + 1].split('/')[0]

            if be_word in words_before_trigger:
                vec_4_training.append(1)
            else:
                vec_4_training.append(0)
            if af_word in words_after_trigger:
                vec_4_training.append(1)
            else:
                vec_4_training.append(0)

            # print sen
            for r in rule:
                match = r.findall(sen)
                if match:
                    vec_4_training.append(1)
                    # zero = False
                else:
                    vec_4_training.append(0)

            # assert 1==0

            if len(vec_4_training) != 19:
                print vec_4_training
                print file_name + ':' + input_sent

            if ADD_FEATURE:
                # Add Auxiliary Feature
                vec_4_training += get_Auxiliary_Feature(words, index, w)
                # Add Syntactic Feature
                try:
                    vec_4_training += get_Syntactic_Feature(sen, index)
                except UnboundLocalError:
                    continue

            # if vec_4_training not in vec_4_training_list:
            vec_4_training_list.append(vec_4_training)


            # decide aux type
            # the same  X X opposite  likewise
            tmp_sen = words + ['none', 'none', 'none']
            next_word = []
            for aux_idx in range(1, 4):
                if '\\' in tmp_sen[index + aux_idx]:
                    next_w = tmp_sen[index + aux_idx].split('/')[0] + tmp_sen[index + aux_idx].split('/')[1]
                else:
                    next_w = tmp_sen[index + aux_idx].split('/')[0]

                next_word.append(next_w)

            if lemma(next_word[0]) == 'so' or lemma(next_word[0]) == 'likewise' or lemma(next_word[1]) == 'same' or lemma(next_word[1]) == 'opposite' or lemma(next_word[2]) == 'opposite':
                aux_type.append('so')
            else:
                aux_type.append(lemma(w))
            trigger.append(w)
            if tag == 'T':
                data_label.append(1)
                # sens_trigger_idx.append(w)
            else:
                data_label.append(0)

                # sens_trigger_idx.append('None')
    # if not check_sen(sen):
    #     print file_name

    return vec_4_training_list, data_label, trigger, aux_type

def filter_aux(data,lable, aux):
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


def get_Auxiliary_Feature(words, index, w):
    avec =[]
    add_feature_aux_word = ''
    add_feature_aux_lemma = ''
    add_feature_aux_type = ''
    tmp_sen = words + ['none', 'none', 'none']
    next_word = []
    for aux_idx in range(1, 4):
        if '\\' in tmp_sen[index + aux_idx]:
            next_w = tmp_sen[index + aux_idx].split('/')[0] + tmp_sen[index + aux_idx].split('/')[1]
        else:
            next_w = tmp_sen[index + aux_idx].split('/')[0]

        next_word.append(next_w)

    if lemma(next_word[0]) == 'so':
        add_feature_aux_type = 'so'
        add_feature_aux_word = 'so'
        add_feature_aux_lemma = lemma('so')
    elif lemma(next_word[0]) == 'likewise':
        add_feature_aux_type = 'likewise'
        add_feature_aux_word = 'likewise'
        add_feature_aux_lemma = lemma('likewise')
    elif lemma(next_word[1]) == 'same':
        add_feature_aux_type = 'same'
        add_feature_aux_word = 'same'
        add_feature_aux_lemma = lemma('same')
    elif lemma(next_word[1]) == 'opposite' or lemma(next_word[2]) == 'opposite':
        add_feature_aux_type = 'opposite'
        add_feature_aux_word = 'opposite'
        add_feature_aux_lemma = 'opposite'
    else:
        add_feature_aux_type = lemma(w)
        add_feature_aux_lemma = lemma(w)
        add_feature_aux_word = lemma(w)

    if add_feature_aux_type in add_feature_aux_type_2_idx:
        avec.append(add_feature_aux_type_2_idx.index(add_feature_aux_type))
    else:
        avec.append(len(add_feature_aux_type_2_idx))
        add_feature_aux_type_2_idx.append(add_feature_aux_type)

    if add_feature_aux_lemma in add_feature_aux_lemma_2_idx:
        avec.append(add_feature_aux_lemma_2_idx.index(add_feature_aux_lemma))
    else:
        avec.append(len(add_feature_aux_lemma_2_idx))
        add_feature_aux_lemma_2_idx.append(add_feature_aux_lemma)

    if add_feature_aux_word in add_feature_aux_word_2_idx:
        avec.append(add_feature_aux_word_2_idx.index(add_feature_aux_word))
    else:
        avec.append(len(add_feature_aux_word_2_idx))
        add_feature_aux_word_2_idx.append(add_feature_aux_word)
    return avec


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def get_Syntactic_Feature(sen, index):
    # sen = "Not only is development of the new company 's initial machine tied directly to Mr. Cray , so is its balance sheet ."
    svec = []
    parse = berkeleyp.parse(sen)
    tree = Tree.fromstring(parse)
    # tree.pretty_print()

    # Syntactic

    aux_word = sen.split()[index]
    # aux_index = 0
    if len(sen.split()) == len(tree.leaves()):
        aux_index = index
    else:
        if tree.leaves().count(aux_word) == 1:
            aux_index = tree.leaves().index(aux_word)
        else:
            indexs = indices(tree.leaves(), aux_word)
            for ix in indexs:
                if ix > index:
                    aux_index = ix
                    break

    # a c-commands a verb
    word_a = list(tree.leaf_treeposition(aux_index))
    word_a.pop()
    word_a.pop()
    ccommand_list = tree[word_a].pos()[1:]
    find_ccommand = False
    a_command_verb_index = 0
    for idx, (ccommand_word, ccommand_pos) in enumerate(ccommand_list):
        if ccommand_pos in verb_list:
            find_ccommand = True
            a_command_verb_index = idx
            break
    if find_ccommand:
        svec.append(1)
    else:
        svec.append(0)

    # a c-commands a verb that comes after it

    if a_command_verb_index == 0:
        svec.append(1)
    else:
        svec.append(0)

    # a verb c-commands a

    find_c_command = False
    sen_word_pos = tree.pos()
    for idx, (word_item, pos_item) in enumerate(sen_word_pos):
        if pos_item in verb_list:
            word_a = list(tree.leaf_treeposition(aux_index))
            word_v = list(tree.leaf_treeposition(idx))
            word_v.pop()
            word_v.pop()
            if set(word_v).issubset(set(word_a)):
                find_c_command = True
                break
    if find_c_command:
        svec.append(1)
    else:
        svec.append(0)

    # a verb locally c-command a

    find_c_command = False
    sen_word_pos = tree.pos()
    for idx, (word_item, pos_item) in enumerate(sen_word_pos):
        if pos_item in verb_list:
            word_a = list(tree.leaf_treeposition(aux_index))
            word_a.pop()
            while (tree[word_a].label() != 'S' and len(word_a) > 0):
                word_a.pop()
            word_v = list(tree.leaf_treeposition(idx))
            word_v.pop()
            while (tree[word_v].label() != 'S' and len(word_v) > 0):
                word_v.pop()
            if word_v == word_a:
                word_a = list(tree.leaf_treeposition(aux_index))
                word_v = list(tree.leaf_treeposition(idx))
                word_v.pop()
                word_v.pop()
                if set(word_v).issubset(set(word_a)):
                    find_c_command = True
                    break
            else:
                continue
    if find_c_command:
        svec.append(1)
    else:
        svec.append(0)

    # a locally c-commands a verb

    find_c_command = False
    sen_word_pos = tree.pos()
    for idx, (word_item, pos_item) in enumerate(sen_word_pos):
        if pos_item in verb_list:
            word_a = list(tree.leaf_treeposition(aux_index))
            word_a.pop()
            while (tree[word_a].label() != 'S' and len(word_a) > 0):
                word_a.pop()
            word_v = list(tree.leaf_treeposition(idx))
            word_v.pop()
            while (tree[word_v].label() != 'S' and len(word_v) > 0):
                word_v.pop()
            if word_v == word_a:
                word_a = list(tree.leaf_treeposition(aux_index))
                word_a.pop()
                word_a.pop()
                word_v = list(tree.leaf_treeposition(idx))
                if set(word_a).issubset(set(word_v)):
                    find_c_command = True
                    break
            else:
                continue
    if find_c_command:
        svec.append(1)
    else:
        svec.append(0)

    # a is c-commanded by "than", "as", or "so"

    tas_list = ['than', 'as', 'so']

    find_c_command = False
    sen_word_pos = tree.pos()
    for idx, (word_item, pos_item) in enumerate(sen_word_pos):
        if word_item in tas_list:
            word_a = list(tree.leaf_treeposition(aux_index))
            word_tas = list(tree.leaf_treeposition(idx))
            word_tas.pop()
            word_tas.pop()
            if set(word_tas).issubset(set(word_a)):
                find_c_command = True
                break
    if find_c_command:
        svec.append(1)
    else:
        svec.append(0)

    # a is preceded by "than", "as", or "so"

    tmp_words = ['none']+sen.split()
    if tmp_words[index] in tas_list:
        svec.append(1)
    else:
        svec.append(0)

    # a is next to punctuation
    punc_list = list(string.punctuation)
    tmp_words = ['none'] + sen.split()+['none']
    if tmp_words[index] in punc_list or tmp_words[index + 2] in punc_list:
        svec.append(1)
    else:
        svec.append(0)

    # the word "to" precedes a

    tmp_words = ['none'] + sen.split()
    if tmp_words[index] == 'to':
        svec.append(1)
    else:
        svec.append(0)

    # a verb immediately follows a

    sen_word_pos = tree.pos()
    if aux_index < len(sen_word_pos)-1:
        if sen_word_pos[aux_index+1][1] in verb_list:
            svec.append(1)
        else:
            svec.append(0)
    else:
        svec.append(0)

    # a is followed by "too" or "the same"

    sen_word_pos = tree.pos()
    sen_word_pos += [('none', 'none'), ('none', 'none')]

    if sen_word_pos[aux_index+1][0] == 'too' or (sen_word_pos[aux_index+1][0] == 'the' and sen_word_pos[aux_index+2][0] == 'same'):
        svec.append(1)
    else:
        svec.append(0)

    return svec


# 获取所有叶子节点的父节点
def get_lead_nodes(tree):
    nodes = []
    for child in tree:
        flag = True
        if isinstance(child, ParentedTree):
            for c in child:
                if isinstance(c, ParentedTree):
                    flag = False
                    break
            if not flag:
                nodes.extend(get_lead_nodes(child))
            else:
                nodes.append(child)
    return nodes

TAG_PATS = ('NP')
def get_flag_node(node, one_nodes):
    while node.parent is not None and isinstance(node.parent(), ParentedTree):
        node = node.parent()
        if node.label() in TAG_PATS:
            lst = get_lead_nodes(node)
            flag = True
            for n in lst:
                if n in one_nodes:
                    flag = False
                    break
            if flag:
                return node
    return None


def find_pattern(nodes, tags):
    s = t = -1
    one_nodes = []
    for i in range(len(tags)):
        if tags[i]==1 and s < 0:
            s = i
        if tags[i]==1:
            t = i
        if tags[i] == 1:
            one_nodes.append(nodes[i])
    step = 1
    while s-step>=0 or t+step<len(tags):
        if s-step>=0:
            ln = get_flag_node(nodes[s-step], one_nodes)
            if ln is not None:
                return ln
        if t+step<len(tags):
            rn = get_flag_node(nodes[t+step], one_nodes)
            if rn is not None:
                return rn
        step += 1
    return None

def solve_tags(tree, tags):
    ante_tags = [1 if (i==1 or i==3) else 0 for i in tags]
    trig_tags = [1 if (i==2 or i==3) else 0 for i in tags]
    nodes = get_lead_nodes(tree)
    ante_node = find_pattern(nodes, ante_tags)
    trig_node = find_pattern(nodes, trig_tags)
    if ante_node == None:
        ante_str = 'None'
    else:
        ante_str = ' '.join(ante_node.leaves())
    if trig_node == None:
        trig_str = 'None'
    else:
        trig_str = ' '.join(trig_node.flatten())
    return ante_str, trig_str

    '''
def description: get NP Relaton feature vec

input: sen_parse_tree , tag display the antecedent and trigger

( (S (CONJP (RB Not) (RB only)) ...

[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...

return: NP Relation vec [1 X 1]

'''


def get_NP_Relation(sen_parse, tag):
    nvec = []
    ante_str, trig_str = solve_tags(ParentedTree.fromstring(sen_parse), tag)
    NP_ant = ante_str.split()
    NP_tri = trig_str.split()
    ant_sum = np.zeros(100)
    for w in NP_ant:
        ant_sum += np.array(ft_model[w])
    a_we_aver = ant_sum / (len(NP_ant) + 0.0)
    tri_sum = np.zeros(100)
    for w in NP_tri:
        tri_sum += np.array(ft_model[w])
    t_we_aver = tri_sum / (len(NP_tri) + 0.0)
    cos_simi = get_sim(a_we_aver, t_we_aver)
    # cos_simi = 1 - spatial.distance.cosine(a_we_aver, t_we_aver)
    nvec.append(cos_simi)
    return nvec

def get_sim(vec_a, vec_b):
    cos_simi = 1 - spatial.distance.cosine(vec_a, vec_b)
    return cos_simi

'''
def description: get Syntactic feature vec

input: sen_with_pos, ant_start, ant_end, tri_pos, sen_parse_tree

[('Not', 'RB'), ('only', 'RB') ...

11 

18

( (S (CONJP (RB Not) (RB only)) ...

return: syntactic vec [55 X 1]

'''


def get_Syntactic(sen_with_pos, ant_start, ant_end, tri_pos, sen_parse_tree, words):
    svec = []
    # if a's first word is an auxiliary
    Auxiliary_list_do = ['do', 'does', 'did', 'doing', 'done']
    Auxiliary_list_be = ['be', 'is', 'am', 'are', 'was', 'were', "'s", 'been', "'re"]
    Auxiliary_list_so = ['so']
    Auxiliary_list_have = ['have', 'has', 'had', 'having', 'haven']
    Auxiliary_list_to = ['to']
    Auxiliary_list_can = ['can', 'could']
    Auxiliary_list_will = ['will', 'would']
    Auxiliary_list_same = ['same']
    Auxiliary_list_should = ['should']
    Auxiliary_list_may = ['may', 'might', 'must', 'likewise', 'opposite']
    Auxiliary_list = Auxiliary_list_do + Auxiliary_list_be + Auxiliary_list_so
    Auxiliary_list += Auxiliary_list_have + Auxiliary_list_to + Auxiliary_list_can
    Auxiliary_list += Auxiliary_list_will + Auxiliary_list_same + Auxiliary_list_should
    Auxiliary_list += Auxiliary_list_may
    if sen_with_pos[ant_start][0].lower() in Auxiliary_list:
        svec.append(1)
    else:
        svec.append(0)

    # if a's head (i.e., first main verb) is an auxiliary
    Verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    find_done = False
    for i in range(ant_start, ant_end + 1):
        if sen_with_pos[i][1] in Verb_list:
            if sen_with_pos[i][0].lower() in Auxiliary_list:
                svec.append(1)
            else:
                svec.append(0)
            find_done = True
            break
    if not find_done:
        svec.append(0)

    # the POS tag of a's first and last word
    # print '$$$$$$$$$$$$$$$'
    # print len(sen_with_pos)
    # print ant_end
    svec.append(pos_tag_idx.index(sen_with_pos[ant_start][1]))
    svec.append(pos_tag_idx.index(sen_with_pos[ant_end][1]))

    # the frequency of each POS tag in the antecedent
    ant_POS = []
    ant = []
    for i in range(ant_start, ant_end + 1):
        ant_POS.append(sen_with_pos[i][1])
        ant.append(sen_with_pos[i][0])
    for i in range(len(pos_tag_idx)):
        svec.append(ant_POS.count(pos_tag_idx[i]))

    # the frequency of each phrase (i.e., NP, VP, ADJP, etc.) in a's sentence and t's sentence
    ant_sen = (' ').join(ant)
    tri_sen = sen_with_pos[tri_pos][0]
    ant_phrase = []
    tri_phrase = []
    sen_phrase = get_phrase(sen_parse_tree)
    phrase_list = ['NP', 'VP', 'ADJP']
    for phra in phrase_list:
        if phra in sen_phrase:
            phra_list = sen_phrase[phra]
            phra_count = 0
            for n in phra_list:
                if n in ant_sen:
                    phra_count += 1
            ant_phrase.append(phra_count)
            phra_count = 0
            for n in phra_list:
                if n in tri_sen:
                    phra_count += 1
            tri_phrase.append(phra_count)
        else:
            ant_phrase.append(0)
            tri_phrase.append(0)

    # if "than", "as", or "so" is between a and t
    words_between_a_t = []
    for i in range(ant_end + 1, tri_pos):
        words_between_a_t.append(sen_with_pos[i][0])

    wbat = (' ').join(words_between_a_t)
    if 'than' in wbat or 'as' in wbat or 'so' in wbat:
        svec.append(1)
    else:
        svec.append(0)

    # if the word before a has the same POS-tag or lemma as t
    wba = sen_with_pos[ant_start - 1]
    tri = sen_with_pos[tri_pos]
    if wba[1] == tri[1] or lemma(wba[0]) == lemma(tri[0]):
        svec.append(1)
    else:
        svec.append(0)

    # if a word in a c-commands a word in t
    find_c_command = False
    try:
        for i in range(ant_start, ant_end):
            word_a = list(sen_parse_tree.leaf_treeposition(i))
            word_a.pop()
            word_t = list(sen_parse_tree.leaf_treeposition(tri_pos))
            word_a.pop()
            if set(word_a).issubset(set(word_t)):
                svec.append(1)
                find_c_command = True
                break
    except IndexError:
        find_c_command = False
    if not find_c_command:
        svec.append(0)

    # if a's first or last word c-commands the trigger
    find_c_command = False
    # print ant_start
    # print len(sen_parse_tree.leaves())
    # print ant_end
    # print sen_parse_tree.leaf_treeposition(len(sen_parse_tree.leaves())-1)
    try:
        for i in [ant_start if ant_start < len(sen_parse_tree.leaves()) else len(sen_parse_tree.leaves()) - 1 , ant_end if ant_end < len(sen_parse_tree.leaves()) else len(sen_parse_tree.leaves()) - 1]:
            word_a = list(sen_parse_tree.leaf_treeposition(i))
            word_a.pop()
            word_t = list(sen_parse_tree.leaf_treeposition(tri_pos))
            while len(sen_parse_tree[word_a]) == 1:
                word_a.pop()
            if set(word_a).issubset(set(word_t)):
                svec.append(1)
                find_c_command = True
                break
    except IndexError:
        find_c_command = False
    if not find_c_command:
        svec.append(0)

    # Be-Do Form: if the lemma of the token preceding a is be and the t's lemma is do
    tpa = sen_with_pos[ant_start - 1][0]
    tri = sen_with_pos[tri_pos][0]
    if lemma(tpa) == 'be' and lemma(tri) == 'do':
        svec.append(1)
    else:
        svec.append(0)

    # Recency: distance between a and t and the distance between the t's nearest VP and a
    svec.append(tri_pos - ant_end)
    sen_all = (' ').join(words)
    vp_start_end =[]
    if 'VP' in sen_phrase:
        vps = sen_phrase['VP']
        for vp in vps:
            if tri not in vp:
                if sen_all.count(vp) == 1:
                    vp_start = sen_all.index(vp)
                    vp_end = vp_start + len(vp)
                    vp_start_end.append((vp_start, vp_end))
        dis_vp_t = 0
        n_v_s = 0
        n_v_e = 0
        for (vs, ve) in vp_start_end:
            if tri_pos < vs:
                if dis_vp_t < vs - tri_pos:
                    n_v_s = vs
                    n_v_e = ve
                    dis_vp_t = vs - tri_pos
            if tri_pos > ve:
                if dis_vp_t < tri_pos - ve:
                    n_v_s = vs
                    n_v_e = ve
                    dis_vp_t = tri_pos - ve
        if n_v_s > ant_end:
            svec.append(n_v_s - ant_end)
        elif n_v_e < ant_start:
            svec.append(ant_start - n_v_e)
        else:
            svec.append(0)
    else:
        svec.append(0)

    # print len(svec)
    return svec

    # Quotation: if t is between quotation marks and similarly for a (abort)


def get_phrase(tree):
    dict = {}
    subtrees = tree.subtrees()
    for subtree in subtrees:
        label = subtree.label()
        if label in ['NP', 'VP', 'ADJP']:
            if label not in dict:
                dict[label] = []
            lst = []
            for w in subtree.leaves():
                if '*' not in w:
                    lst.append(w)
            dict[label].append(' '.join(lst))
    lst = []
    for w in tree.leaves():
        if '*' not in w:
            lst.append(w)
    dict['S'] = ' '.join(lst)
    return dict

'''
def description: get Matching feature vec

input: sentence with POS tag, the start index of antecedent and the index of trigger

[('Not', 'RB'), ('only', 'RB') ...

11 

18

return: the Matching feature vector [12 X 1]

[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

'''


def get_Matching(sen_with_pos, ant_start, tri_pos):
    mvec = []

    # to prevent ant_start - 3 lower than 0
    none_vec = []
    for i in range(3):
        none_vec.append(('None', 'None'))
    sen_with_pos = none_vec + sen_with_pos
    ant_start += 3
    tri_pos += 3

    # two token window before antecedent and trigger
    ttb_ant = [sen_with_pos[ant_start - 2], sen_with_pos[ant_start - 1]]
    ttb_tri = [sen_with_pos[tri_pos - 2], sen_with_pos[tri_pos - 1]]

    # POS-tags
    pos_2_ant = ttb_ant[0][1]
    pos_1_ant = ttb_ant[1][1]
    pos_2_tri = ttb_tri[0][1]
    pos_1_tri = ttb_tri[1][1]
    if pos_2_ant == pos_2_tri and pos_1_ant == pos_1_tri:
        mvec.append(1)
    else:
        mvec.append(0)
    # lemmas
    lemma_2_ant = lemma(ttb_ant[0][0])
    lemma_1_ant = lemma(ttb_ant[1][0])
    lemma_2_tri = lemma(ttb_tri[0][0])
    lemma_1_tri = lemma(ttb_tri[1][0])

    if lemma_2_ant == lemma_2_tri and lemma_1_ant == lemma_1_tri:
        mvec.append(1)
    else:
        mvec.append(0)
    # words
    word_2_ant = ttb_ant[0][0]
    word_1_ant = ttb_ant[1][0]
    word_2_tri = ttb_tri[0][0]
    word_1_tri = ttb_tri[1][0]
    if word_2_ant == word_2_tri and word_1_ant == word_1_tri:
        mvec.append(1)
    else:
        mvec.append(0)

    # get i th token before antecedent and i-1 th token
    # before trigger POS-tag, lemma and word matching

    match_cases = []
    match_cases.append((sen_with_pos[ant_start - 3], sen_with_pos[tri_pos - 2]))
    match_cases.append((sen_with_pos[ant_start - 2], sen_with_pos[tri_pos - 1]))
    match_cases.append((sen_with_pos[ant_start - 1], sen_with_pos[tri_pos]))

    for i in range(3):
        (ant, tri) = match_cases[i]
        # POS-tag
        if ant[1] == tri[1]:
            mvec.append(1)
        else:
            mvec.append(0)
        # lemma
        if lemma(ant[0]) == lemma(tri[0]):
            mvec.append(1)
        else:
            mvec.append(0)
        # word
        if ant[0] == tri[0]:
            mvec.append(1)
        else:
            mvec.append(0)

    return mvec





def get_syntactic_feature(VERB_LIST, _tmp_sen, idx):

        # sen = "Not only is development of the new company 's initial machine tied directly to Mr. Cray , so is its balance sheet ."
        svec = []
        # _tmp_sen = current_sentence.sen
        _parse = berkeleyp.parse(_tmp_sen)
        # current_sentence.sen_parse = _parse
        _tree = Tree.fromstring(_parse)
        # tree.pretty_print()

        # Syntactic
        aux_word = _tmp_sen.split()[idx]
        # aux_index = 0
        if len(_tmp_sen.split()) == len(_tree.leaves()):
            aux_index = idx
        else:
            if _tree.leaves().count(aux_word) == 1:
                aux_index = _tree.leaves().index(aux_word)
            else:
                indexs = indices(_tree.leaves(), aux_word)
                for ix in indexs:
                    if ix > idx:
                        aux_index = ix
                        break

        # a c-commands a verb
        word_a = list(_tree.leaf_treeposition(aux_index))
        word_a.pop()
        word_a.pop()
        ccommand_list = _tree[word_a].pos()[1:]
        find_ccommand = False
        a_command_verb_index = 0
        for idx, (ccommand_word, ccommand_pos) in enumerate(ccommand_list):
            if ccommand_pos in VERB_LIST:
                find_ccommand = True
                a_command_verb_index = idx
                break
        if find_ccommand:
            svec.append(1)
        else:
            svec.append(0)

        # a c-commands a verb that comes after it
        if a_command_verb_index == 0:
            svec.append(1)
        else:
            svec.append(0)

        # a verb c-commands a

        find_c_command = False
        sen_word_pos = _tree.pos()
        for idx, (word_item, pos_item) in enumerate(sen_word_pos):
            if pos_item in VERB_LIST:
                word_a = list(_tree.leaf_treeposition(aux_index))
                word_v = list(_tree.leaf_treeposition(idx))
                word_v.pop()
                word_v.pop()
                if set(word_v).issubset(set(word_a)):
                    find_c_command = True
                    break
        if find_c_command:
            svec.append(1)
        else:
            svec.append(0)

        # a verb locally c-command a

        find_c_command = False
        sen_word_pos = _tree.pos()
        for idx, (word_item, pos_item) in enumerate(sen_word_pos):
            if pos_item in VERB_LIST:
                word_a = list(_tree.leaf_treeposition(aux_index))
                word_a.pop()
                while (_tree[word_a].label() != 'S' and len(word_a) > 0):
                    word_a.pop()
                word_v = list(_tree.leaf_treeposition(idx))
                word_v.pop()
                while (_tree[word_v].label() != 'S' and len(word_v) > 0):
                    word_v.pop()
                if word_v == word_a:
                    word_a = list(_tree.leaf_treeposition(aux_index))
                    word_v = list(_tree.leaf_treeposition(idx))
                    word_v.pop()
                    word_v.pop()
                    if set(word_v).issubset(set(word_a)):
                        find_c_command = True
                        break
                else:
                    continue
        if find_c_command:
            svec.append(1)
        else:
            svec.append(0)

        # a locally c-commands a verb

        find_c_command = False
        sen_word_pos = _tree.pos()
        for idx, (word_item, pos_item) in enumerate(sen_word_pos):
            if pos_item in VERB_LIST:
                word_a = list(_tree.leaf_treeposition(aux_index))
                word_a.pop()
                while (_tree[word_a].label() != 'S' and len(word_a) > 0):
                    word_a.pop()
                word_v = list(_tree.leaf_treeposition(idx))
                word_v.pop()
                while (_tree[word_v].label() != 'S' and len(word_v) > 0):
                    word_v.pop()
                if word_v == word_a:
                    word_a = list(_tree.leaf_treeposition(aux_index))
                    word_a.pop()
                    word_a.pop()
                    word_v = list(_tree.leaf_treeposition(idx))
                    if set(word_a).issubset(set(word_v)):
                        find_c_command = True
                        break
                else:
                    continue
        if find_c_command:
            svec.append(1)
        else:
            svec.append(0)

        # a is c-commanded by "than", "as", or "so"

        tas_list = ['than', 'as', 'so']

        find_c_command = False
        sen_word_pos = _tree.pos()
        for idx, (word_item, pos_item) in enumerate(sen_word_pos):
            if word_item in tas_list:
                word_a = list(_tree.leaf_treeposition(aux_index))
                word_tas = list(_tree.leaf_treeposition(idx))
                word_tas.pop()
                word_tas.pop()
                if set(word_tas).issubset(set(word_a)):
                    find_c_command = True
                    break
        if find_c_command:
            svec.append(1)
        else:
            svec.append(0)

        # a is preceded by "than", "as", or "so"

        tmp_words = ['none']+_tmp_sen.split()
        if tmp_words[idx] in tas_list:
            svec.append(1)
        else:
            svec.append(0)

        # a is next to punctuation
        punc_list = list(string.punctuation)
        tmp_words = ['none'] + _tmp_sen.split()+['none']
        if tmp_words[idx] in punc_list or tmp_words[idx + 2] in punc_list:
            svec.append(1)
        else:
            svec.append(0)

        # the word "to" precedes a

        tmp_words = ['none'] + _tmp_sen.split()
        if tmp_words[idx] == 'to':
            svec.append(1)
        else:
            svec.append(0)

        # a verb immediately follows a

        sen_word_pos = _tree.pos()
        if aux_index < len(sen_word_pos)-1:
            if sen_word_pos[aux_index+1][1] in VERB_LIST:
                svec.append(1)
            else:
                svec.append(0)
        else:
            svec.append(0)

        # a is followed by "too" or "the same"

        sen_word_pos = _tree.pos()
        sen_word_pos += [('none', 'none'), ('none', 'none')]

        if sen_word_pos[aux_index+1][0] == 'too' or (sen_word_pos[aux_index+1][0] == 'the' and sen_word_pos[aux_index+2][0] == 'same'):
            svec.append(1)
        else:
            svec.append(0)

        return svec, _parse





def get_feature_vec(sen_parse, tag, words):

    tree = Tree.fromstring(sen_parse)
    # tree.pretty_print()

    ante_tags = [1 if (int(i) == 1 or int(i) == 3) else 0 for i in tag]
    trig_tags = [1 if (int(i) == 2 or int(i) == 3) else 0 for i in tag]
    i = 0
    # print ante_tags
    while (ante_tags[i] != 1):
        # print i
        i += 1
    antecedent_start = i
    while (ante_tags[i] == 1 and i + 1 < len(ante_tags)):
        i += 1
    antecedent_end = i - 1
    trigger_pos = trig_tags.index(1)
    pos_tag =nltk.pos_tag(word_tokenize((' ').join(words)))
    mvec = get_Matching(pos_tag, antecedent_start, trigger_pos)
    # print sen_parse
    # print tag
    nvec = get_NP_Relation(sen_parse, tag)
    svec = get_Syntactic(pos_tag, antecedent_start, antecedent_end, trigger_pos, tree, words)
    feature_vec = mvec + nvec + svec
    assert len(feature_vec) == 68
    return feature_vec


def read_file(file):
    with open(file) as fin:
        tree_lst = []
        str_lst = []
        count = 0
        while True:
            line = fin.readline()
            if not line:
                break
            if line[0] == '(':
                if count != 0:
                    tree_lst.append(Tree.fromstring(''.join(str_lst)))
                    del str_lst[:]
                count += 1
                str_lst.append(line)
            elif line != '\n':
                str_lst.append(line)
        tree_lst.append(Tree.fromstring(''.join(str_lst)))
        return tree_lst

PATTERNS = ['ADJP', 'VP']
def get_dict(tree):
    dict = {}
    subtrees = tree.subtrees()
    for subtree in subtrees:
        label = subtree.label()
        if label in PATTERNS:
            if label not in dict:
                dict[label] = []
            lst = []
            for w in subtree.leaves():
                if '*' not in w:
                    lst.append(w)
            dict[label].append(' '.join(lst))
    lst = []
    for w in tree.leaves():
        if '*' not in w:
            lst.append(w)
    dict['S'] = ' '.join(lst)
    return dict


def get_antecedent(str):
    # str = '(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))'
    # str = '( (S (NP (DT the) (NN dog)) (VP (VBZ chases) (NP (DT the) (NN cat)))) )'
    # str = "( (S (S (S (VP (VBN Hard) (PP (IN on) (NP (NP (NP (DT the) (NNS heels)) (PP (IN of) (NP (NP (NNP Friday) (POS 's)) (JJ 190-point) (NN stock-market) (NN plunge)))) (CC and) (NP (NP (DT the) (NN uncertainty)) (SBAR (WHNP (WDT that)) (S (VP (VBZ 's) (VP (VBN followed)))))))))) (, ,) (NP (DT a) (JJ few) (JJ big) (NN brokerage) (NNS firms)) (VP (VBP are) (VP (VBG rolling) (PRT (RP out)) (NP (NP (JJ new) (NNS ads)) (VP (VBG trumpeting) (NP (DT a) (JJ familiar) (NN message))))))) (: :) (S (S (VP (VB Keep) (PP (IN on) (NP (NN investing))))) (, ,) (NP (DT the) (NN market)) (VP (VBZ 's) (ADVP (RB just)) (ADJP (JJ fine)))) (. .) (NP (PRP$ Their) (NN mission)) (VP (VBZ is) (S (VP (TO to) (VP (VB keep) (NP (NNS clients)) (PP (IN from) (S (VP (VBG fleeing) (NP (DT the) (NN market))))) (, ,) (SBAR (IN as) (S (NP (JJ individual) (NNS investors)) (VP (VBD did) (PP (IN in) (NP (NNS droves))) (PP (IN after) (NP (NP (DT the) (NN crash)) (PP (IN in) (NP (NNP October))))))))))))) )"
    tree = Tree.fromstring(str)
    # tree.pretty_print()
    res = get_dict(tree)
    return res


def file_test(filename):
    dict_ = []
    tree_lst = read_file(filename)
    for tree in tree_lst:
        # print(get_dict(tree))
        tree.pretty_print()

        dict_.append(get_dict(tree))
    return dict_


def save_dict(dict, filename):
    with open(filename, 'wb')as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




def split_out_antecedent(sen, sen_parse, truth_label, trigger_idx):
    # sen = line.replace('\n', '')
    words = sen.split()
    parser_tree = sen_parse
    sent_label = truth_label
    trigger = trigger_idx
    
    g_ant = ''  # gold
    b_ant = ''  # before
    a_ant = ''  # after
    for lable_idx, lable in enumerate(sent_label):
        if int(lable):
            g_ant += words[lable_idx]
            if lable_idx != len(sent_label) - 1:
                g_ant += ' '

    if len(sen.split(g_ant)) == 2:
        b_ant = sen.split(g_ant)[0]
        a_ant = sen.split(g_ant)[1]
    else:
        ant_idx = 0
        while (int(sent_label[ant_idx]) == 0):
            b_ant += words[ant_idx]
            if int(sent_label[ant_idx + 1]) == 0:
                b_ant += ' '
            ant_idx += 1
        while (int(sent_label[ant_idx]) == 0):
            ant_idx += 1
        while ant_idx < len(words):
            a_ant += words[ant_idx]
            if ant_idx != len(words)-1:
                a_ant += ' '
            ant_idx += 1

    antecedent = []
    before_ant = []
    after_ant = []
    tag_4_data = []
    label_4_data = []

    antecedent.append(g_ant)
    if not b_ant:
        b_ant = ' '
    if not a_ant:
        a_ant = ' '
    before_ant.append(b_ant)
    after_ant.append(a_ant)
    tag_4_data.append('1')
    tmp = sent_label
    tmp[int(trigger)] = '2'
    label_4_data.append(tmp)


    parse_dict = get_antecedent(parser_tree)
    if 'VP' in parse_dict:
        vps = parse_dict['VP']
        # print vps
        for v in vps:
            if v == g_ant or v + ' ' == g_ant:
                continue
            else:
                if len(sen.split(v)) == 2:
                    antecedent.append(v)
                    # print sen.split(v)
                    before_ant.append(sen.split(v)[0])
                    tmp_label = []
                    for o in range(len(sen.split(v)[0].split())):
                        tmp_label.append('0')
                    for o in range(len(v.split())):
                        tmp_label.append('1')
                    for o in range(len(sen.split(v)[1].split())):
                        tmp_label.append('0')
                    if tmp_label[int(trigger)] == '1':
                        tmp_label[int(trigger)] = '3'
                    else:
                        tmp_label[int(trigger)] = '2'
                    label_4_data.append(tmp_label)
                    after_ant.append(sen.split(v)[1])
                    tag_4_data.append('0')

    if 'ADJP' in parse_dict:
        adjs = parse_dict['ADJP']
        # print adjs
        for adj in adjs:
            if adj == g_ant or adj + ' ' == g_ant:
                continue
            else:
                if len(sen.split(adj)) == 2:
                    antecedent.append(adj)
                    before_ant.append(sen.split(adj)[0])
                    after_ant.append(sen.split(adj)[1])
                    tag_4_data.append('0')
                    tmp_label = []
                    for o in range(len(sen.split(adj)[0].split())):
                        tmp_label.append('0')
                    for o in range(len(adj.split())):
                        tmp_label.append('1')
                    for o in range(len(sen.split(adj)[1].split())):
                        tmp_label.append('0')
                    if tmp_label[int(trigger)] == '1':
                        tmp_label[int(trigger)] = '3'
                    else:
                        tmp_label[int(trigger)] = '2'
                    label_4_data.append(tmp_label)

    return antecedent, before_ant, after_ant, tag_4_data, label_4_data


def get_sum_pooling_vec(sen):
    # sen = "Not only is development of the new company 's initial machine tied directly to Mr. Cray , so is its balance sheet . "
    words = sen.split()
    sum_pooling = np.zeros(100)
    for w in words:
        sum_pooling += np.array(ft_model[w])
    # print len(sum_pooling)
    return sum_pooling
