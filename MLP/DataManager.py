# coding = utf-8

import copy
import re
from pattern.en import lemma
from nltk.tree import Tree
from tools import *
import string
import random
import cPickle as pickle
import fasttext
import numpy as np
import torch
import os
import argparse
from Sentence import Sentence

# jar_path = "/Users/liuyuanxing/Code/Leeeeo_Toolkit/python/berkeleyparser/BerkeleyParser-1.7.jar"
# gr_path = "/Users/liuyuanxing/Code/Leeeeo_Toolkit/python/berkeleyparser/eng_sm6.gr"

jar_path = "/users4/yxliu/Leeeeo_toolkit/berkeleyparser/BerkeleyParser-1.7.jar"
gr_path = "/users4/yxliu/Leeeeo_toolkit/berkeleyparser/eng_sm6.gr"


from BerkeleyParser import parser
berkeleyp = parser(jar_path, gr_path)

class DataManager(object):
    def __init__(self, args):

        self.data_files = ['wsj.section{}'.format(
            i if i > 9 else '0{}'.format(i)) for i in range(25)]
        self.data_root_path = './datas/wsj_tagged/'
        # whether add feature
        self.add_feature = args['feature']
        # whehter use train-test
        self.train_test = args['train_test']
        self.overwrite = args['overwrite']
        self.train_files = ['wsj.section{}'.format(
            i.__str__() if i > 9 else '0' + i.__str__()) for i in range(20)]
        self.test_files = ['wsj.section{}'.format(
            i.__str__() if i > 9 else '0' + i.__str__()) for i in range(20, 25)]
        self.sentences = []
        self.trigger_generate_sentences = []

        self.antecedent_generate_sentences = []

        # for get_auxiliary_feature
        self.aux_word_2_idx = []
        self.aux_lemma_2_idx = []

        # for antecedent
        self.fasttext_model = fasttext.load_model('./datas/fasttext_model_4_all_wsj.bin')

    def init(self):
        self.load_tagged_data()
        self.process_data()
        self.process_data_4_antecedent()

    def load_tagged_data(self):
        _path_to_self_sentences = './datas/data_manager/self_sentences.pkl'
        if os.path.exists(_path_to_self_sentences) and self.overwrite == False:
            print('found preprocessed file')
            print('load self.sentences from {}'.format(_path_to_self_sentences))
            with open(_path_to_self_sentences, 'r') as f:
                self.sentences = pickle.load(f)
        else:
            print('start loading tagged data')
            for data_file in self.data_files:
                print('processing {}'.format(data_file))
                with open(self.data_root_path + data_file, 'r') as f:
                    for line in f:
                        line = line.replace('\xef\xbb\xbf', '').replace(
                            '\t', '').replace('\r', '').replace('\n', '')
                        if line:
                            self.sentences.append(Sentence(line, data_file))
            print('finish loading tagged data')
            with open(_path_to_self_sentences, 'w') as f:
                pickle.dump(self.sentences, f,protocol=pickle.HIGHEST_PROTOCOL)

    def process_data(self):
        """ get sentence info """
        
        # _path_to_self_trigger_generate_sentences = './datas/data_manager/self_trigger_generate_sentences.pkl'
        _path_to_self_trigger_generate_sentences = './datas/data_manager/ete_trigger_generate_sentences.pkl'
        
        if os.path.exists(_path_to_self_trigger_generate_sentences) and self.overwrite == False:
            print('found preprocessed file')
            print('load self.trigger_generate_sentences from {}'.format(_path_to_self_trigger_generate_sentences))
            with open(_path_to_self_trigger_generate_sentences, 'r') as f:
                self.trigger_generate_sentences = pickle.load(f)
        else:
            print('start processing sentences')
            for idx, sentence in enumerate(self.sentences):
                print('processing {}/{}'.format(idx, len(self.sentences)))
                _words = sentence.sen_tagged.split()
                # get sen from tagged sentence, 7\/8/CD/O in the sentence
                sentence.sen = ' '.join([_word.split('/')[0] + _word.split('/')[1] if '\\' in _word else _word.split('/')[0] for _word in _words])
                sentence.words_list = [_word.split('/')[0] + _word.split('/')[1] if '\\' in _word else _word.split('/')[0] for _word in _words]
                sentence.pos_list = [POS_2_INDEX.index(_word.split('/')[2])  if '\\' in _word else POS_2_INDEX.index(_word.split('/')[1]) for _word in _words]
                try:
                    sentence.tag_list = [TAG_2_INDEX.index(_word.split('/')[3])  if '\\' in _word else TAG_2_INDEX.index(_word.split('/')[2]) for _word in _words]
                except ValueError as e:
                    print e.message
                    print(_words)
                    print(sentence.wsj_section)
                    exit(0)

                sentence.truth_label = [1 if tag_idx != 2 and tag_idx != 5 else 0 for tag_idx in sentence.tag_list]
                _neg_trigger_list = []
                for index, word in enumerate(_words):
                    # if word:
                    # 7\/8/CD/O
                    current_sentence = copy.deepcopy(sentence)
                    current_sentence.is_generated = True
                    _pos = POS_2_INDEX[current_sentence.pos_list[index]]
                    if _pos  in COMMON_TRIGGER_POS_LIST:
                        vec_4_training = [1 if current_sentence.words_list[index] in COMMON_TRIGGER_LIST else 0]
                        # append (the pos index of 3 words before current word, pos index of current word, the pos index of 3 words after current word) 
                        # into training vector, then divided by the length of pos index list
                        _tmp_pos_list = [-1, -1, -1] + current_sentence.pos_list + [-1, -1, -1]
                        _selected_pos_list = _tmp_pos_list[index:index+7]
                        _selected_pos_idx = [_pos_idx/float(POS_2_INDEX_LENGTH) for _pos_idx in _selected_pos_list]
                        vec_4_training.extend(_selected_pos_idx)

                        _tmp_words_list = ['none'] + current_sentence.words_list + ['none']
                        _selected_words_list = _tmp_words_list[index:index+3]
                        _be_word = _selected_words_list[0]
                        vec_4_training.append(1 if _be_word in WORDS_BEFORE_TRIGGER else 0)
                        _af_word = _selected_words_list[1]
                        vec_4_training.append(1 if _af_word in WORDS_AFTER_TRIGGER else 0)

                        for _rule in RULE:
                            _match = _rule.findall(current_sentence.sen)
                            vec_4_training.append(1 if _match else 0)

                        if len(vec_4_training) != 19:
                            print vec_4_training
                            print current_sentence.file_name + ':' + current_sentence.tagged
                        current_sentence.sen_vec = copy.deepcopy(vec_4_training) 
                        # Add Auxiliary Feature
                        vec_4_training.extend(self.get_auxiliary_feature(current_sentence, index))
                        # Add Syntactic Feature
                        try:
                            _tmp, _parse  = get_syntactic_feature(VERB_LIST, current_sentence.sen, index)
                            current_sentence.sen_parse = _parse
                            vec_4_training += _tmp
                        except UnboundLocalError:
                            continue

                        current_sentence.sen_vec_feature = copy.deepcopy(vec_4_training)
                        current_sentence.trigger_label = 1 if current_sentence.tag_list[index] == 5 else 0
                        if current_sentence.trigger_label == 0:
                            _neg_trigger_list.append(current_sentence)
                        else:
                            self.trigger_generate_sentences.append(current_sentence)
                random.shuffle(_neg_trigger_list)
                try:
                    self.trigger_generate_sentences.append(_neg_trigger_list[0])
                except IndexError:
                    pass
                del _neg_trigger_list
            print('finish processing sentences')
            with open(_path_to_self_trigger_generate_sentences, 'w') as f:
                pickle.dump(self.trigger_generate_sentences, f,protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_auxiliary_feature(self, current_sentence, idx):
        avec =[]
        _aux_type = None
        _tmp_sen = current_sentence.words_list + ['none', 'none', 'none']
        _current_word = _tmp_sen[idx]
        _next_words = _tmp_sen[idx+1:idx+4]
        _next_words = [lemma(_next_word) for _next_word in _next_words]
        try:
            avec.append(self.aux_word_2_idx.index(_current_word))
        except ValueError:
            avec.append(len(self.aux_word_2_idx))
            self.aux_word_2_idx.append(_current_word)
        try:
            avec.append(self.aux_lemma_2_idx.index(lemma(_current_word)))
        except ValueError:
            avec.append(len(self.aux_lemma_2_idx))
            self.aux_lemma_2_idx.append(lemma(_current_word))
        if _next_words[0] == 'so' or _next_words[0] == 'likewise' or _next_words[1] == 'same' or _next_words[1] == 'opposite' or _next_words[2] == 'opposite':
            _aux_type = 'so'
        else:
            _aux_type = 'modal' if lemma(_current_word) not in AUXILIARY_TYPE else lemma(_current_word)

        avec.append(AUXILIARY_TYPE.index(_aux_type))
        current_sentence.aux_type = _aux_type
        current_sentence.trigger = _current_word
        current_sentence.trigger_index = idx
        return avec

    def get_opennmt_tensor(self):
        if not os.path.exists('./datas/data_4_OpenNMT/sen_be') or self.overwrite:
            print 'writing files to ./datas/data_4_OpenNMT'
            be_sens  = [ _sentence.words_before_antecedent for _sentence in self.antecedent_generate_sentences]
            at_sens  = [ _sentence.antecedent for _sentence in self.antecedent_generate_sentences]
            af_sens  = [ _sentence.words_after_antecedent for _sentence in self.antecedent_generate_sentences]
            with open('./datas/data_4_OpenNMT/sen_be','w') as f:
                f.write('\n'.join(be_sens))
            
            with open('./datas/data_4_OpenNMT/sen_at','w') as f:
                f.write('\n'.join(at_sens))

            with open('./datas/data_4_OpenNMT/sen_af','w') as f:
                f.write('\n'.join(af_sens))
        else:
            print('find ./datas/data_4_OpenNMT/sen_be skip')

        if not os.path.exists('./datas/data_4_OpenNMT/sen_be_hidden') or self.overwrite:
            print 'running translate.py to get hidden and attention tensor'
            _path_list = ['./datas/data_4_OpenNMT/sen_be','./datas/data_4_OpenNMT/sen_at','./datas/data_4_OpenNMT/sen_af']
            for _path in _path_list:
                _cmd = 'python2.7 ./OpenNMT-py/translate.py -model ./datas/data_4_OpenNMT/encoder_model.pt -src {} -replace_unk -verbose -gpu 0'.format(_path)
                os.system(_cmd)
        else:
            print('find ./datas/data_4_OpenNMT/sen_be_hidden skip tranlate')

        # load hidden tensor
        with open('./datas/data_4_OpenNMT/sen_be_hidden', 'r') as f:
            _tmp_be_hidden_tensor = torch.load(f)

        with open('./datas/data_4_OpenNMT/sen_at_hidden', 'r') as f:
            _tmp_at_hidden_tensor = torch.load(f)

        with open('./datas/data_4_OpenNMT/sen_af_hidden', 'r') as f:
            _tmp_af_hidden_tensor = torch.load(f)


        # load attention tensor
        with open('./datas/data_4_OpenNMT/sen_be_attention', 'r') as f:
            _tmp_be_attention_tensor = torch.load(f)

        with open('./datas/data_4_OpenNMT/sen_at_attention', 'r') as f:
            _tmp_at_attention_tensor = torch.load(f)

        with open('./datas/data_4_OpenNMT/sen_af_attention', 'r') as f:
            _tmp_af_attention_tensor = torch.load(f)

        print 'loading tensor the sentence'
        _new_antecedent_sentence_list = []
        for idx, sentence in enumerate(self.antecedent_generate_sentences):
            print('loading vector: {}/{}'.format(idx, len(self.antecedent_generate_sentences)))
            _sentence = copy.deepcopy(sentence)
            _sentence.before_antecedent_hidden_tensor = _tmp_be_hidden_tensor[idx]
            _sentence.antecedent_hidden_tensor = _tmp_at_hidden_tensor[idx]
            _sentence.after_antecedent_hidden_tensor = _tmp_af_hidden_tensor[idx]
            _sentence.before_antecedent_attention_tensor = _tmp_be_attention_tensor[idx]
            _sentence.antecedent_attention_tensor = _tmp_at_attention_tensor[idx]
            _sentence.after_antecedent_attention_tensor = _tmp_af_attention_tensor[idx]
            _new_antecedent_sentence_list.append(_sentence)
        self.antecedent_generate_sentences = _new_antecedent_sentence_list
        del _new_antecedent_sentence_list




    def process_data_4_antecedent(self):
        # _path_to_self_antecedent_generate_sentences = './datas/data_manager/self_antecedent_generate_sentences.pkl'
        _path_to_self_antecedent_generate_sentences = './datas/data_manager/ete_antecedent_generate_sentences.pkl'
        
        print('start processing sentences for antecedent')
        print('split sentence into before_ant, antecedent, after_ant, tag_4_data, label_4_data')
        _new_antecedent_sentence_list = []        
        for idx, sentence in enumerate(self.trigger_generate_sentences):
            print('split {}/{}'.format(idx, len(self.trigger_generate_sentences)))
            if sentence.trigger_label:
                antecedent, before_ant, after_ant, tag_4_data, label_4_data = split_out_antecedent(
                    sentence.sen, sentence.sen_parse, sentence.truth_label, sentence.trigger_index)
                _tmp_neg_sentence_list = []
                for b, a, af, t, l in zip(before_ant, antecedent, after_ant, tag_4_data, label_4_data):
                    if '1' not in l and 1 not in l  and '3' not in l and 3 not in l:
                        continue
                    else:
                        current_sentence = copy.deepcopy(sentence)
                        current_sentence.words_before_antecedent = b if len(b)>1 else 'None'
                        current_sentence.antecedent = a if len(a)>1 else 'None'
                        current_sentence.words_after_antecedent = af if len(af)>1 else 'None'
                        # current_sentence.antecedent_label = t if t else 'None'
                        current_sentence.antecedent_label = int(t)
                        current_sentence.sen_tag_label = l if l else 'None'
                        
                        if current_sentence.antecedent_label == 1:
                            _new_antecedent_sentence_list.append(current_sentence)
                        else:
                            _tmp_neg_sentence_list.append(current_sentence)
            if len(_tmp_neg_sentence_list)>0:
                random.shuffle(_tmp_neg_sentence_list)
                _new_antecedent_sentence_list.append(_tmp_neg_sentence_list[0])

        self.antecedent_generate_sentences = _new_antecedent_sentence_list
        del _new_antecedent_sentence_list
        print('get openNMT hidden and attention')
        self.get_opennmt_tensor()

        _new_antecedent_sentence_list = []
        print('generate antecedent input vector')
        for idx, _sentence in enumerate(self.antecedent_generate_sentences):
            print('generate vector: {}/{}'.format(idx, len(self.antecedent_generate_sentences)))
            sentence = copy.deepcopy(_sentence)
            words = sentence.words_list
            trigger_vec = self.fasttext_model[sentence.trigger]
            fvec = get_feature_vec(sentence.sen_parse, sentence.sen_tag_label, words)
            tri_vec = trigger_vec + fvec
            # tri_vec.append(trigger_vec + fvec)

            # self.input_vec_sum = None
            trigger_vec_tensor = torch.from_numpy(np.array(trigger_vec)).float()
            ba_sum_vec = torch.from_numpy(get_sum_pooling_vec(sentence.words_before_antecedent)).float()
            ant_sum_vec = torch.from_numpy(get_sum_pooling_vec(sentence.antecedent)).float()
            aa_sum_vec = torch.from_numpy(get_sum_pooling_vec(sentence.words_after_antecedent)).float()
            sentence.input_vec_sum = torch.cat((ant_sum_vec,ba_sum_vec,aa_sum_vec ,trigger_vec_tensor)).numpy().tolist()

            # self.input_vec_sum_feature = None
            sentence.input_vec_sum_feature = torch.cat(
                (ant_sum_vec, ba_sum_vec, aa_sum_vec, trigger_vec_tensor, torch.from_numpy(np.array(fvec)).float())).numpy().tolist()

            # self.input_vec_hidden = None
            tri_vec = trigger_vec
            tri = torch.from_numpy(np.array(tri_vec)).float()
            
            _input_vector = torch.cat((sentence.antecedent_hidden_tensor.float(), sentence.before_antecedent_hidden_tensor.float()))
            _input_vector2 = torch.cat((sentence.after_antecedent_hidden_tensor.float(), tri))
            sentence.input_vec_hidden = torch.cat((_input_vector, _input_vector2)).numpy().tolist()

            # self.input_vec_hidden_feature = None
            tri_vec = trigger_vec + fvec
            tri = torch.from_numpy(np.array(tri_vec)).float()
            # _input_vector = torch.cat((sentence.antecedent_hidden_tensor.float(), sentence.before_antecedent_hidden_tensor.float()), 1)
            _input_vector2 = torch.cat((sentence.after_antecedent_hidden_tensor.float(), tri))
            sentence.input_vec_hidden_feature = torch.cat((_input_vector, _input_vector2)).numpy().tolist()

            # self.input_vec_attention = None
            tri_vec = trigger_vec
            tri = torch.from_numpy(np.array(tri_vec)).float()
            _input_vector = torch.cat((sentence.antecedent_attention_tensor.float(), sentence.before_antecedent_attention_tensor.float()))
            _input_vector2 = torch.cat((sentence.after_antecedent_attention_tensor.float(), tri))
            sentence.input_vec_attention = torch.cat((_input_vector, _input_vector2)).numpy().tolist()
            
            # self.input_vec_attention_feature = None
            tri_vec = trigger_vec + fvec
            tri = torch.from_numpy(np.array(tri_vec)).float()
            # _input_vector = torch.cat((sentence.antecedent_attention_tensor.float(), sentence.before_antecedent_attention_tensor.float()), 1)
            _input_vector2 = torch.cat((sentence.after_antecedent_attention_tensor.float(), tri))
            sentence.input_vec_attention_feature = torch.cat((_input_vector, _input_vector2)).numpy().tolist()
            # GRU hidden, Tensor, Size: sentences_length X 6000
            del sentence.before_antecedent_hidden_tensor
            del sentence.antecedent_hidden_tensor
            del sentence.after_antecedent_hidden_tensor
            # attention
            del sentence.before_antecedent_attention_tensor
            del sentence.antecedent_attention_tensor
            del sentence.after_antecedent_attention_tensor
            _new_antecedent_sentence_list.append(sentence)
        self.antecedent_generate_sentences = _new_antecedent_sentence_list
        del _new_antecedent_sentence_list

        print('finish processing sentences for antecedent')
        with open(_path_to_self_antecedent_generate_sentences, 'w') as f:
            pickle.dump(self.antecedent_generate_sentences, f,protocol=pickle.HIGHEST_PROTOCOL)


