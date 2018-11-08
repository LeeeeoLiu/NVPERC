# coding=utf-8
import pickle as pickle

if __name__ == '__main__':
    ROOT_DATA_PATH = '../data_manager/'

    with open('{}self_trigger_generate_sentences.pkl'.format(ROOT_DATA_PATH), 'rb') as f:
        # get all the sentences for trigger detection
        _sentences = pickle.load(f)

    prep_trigger_data = '../prep_trigger_data/'

    sen_vec = 'sen_vec'
    sen_vec_feature = 'sen_vec_feature'
    trigger_label = 'trigger_label'
    trigger = 'trigger'
    aux_type = 'aux_type'
    sen = 'sen'

    input_sen_vec = open(prep_trigger_data + sen_vec + '.txt', 'w+')
    input_sen_vec_feature = open(prep_trigger_data + sen_vec_feature + '.txt', 'w+')
    input_trigger_label = open(prep_trigger_data + trigger_label + '.txt', 'w+')
    input_trigger = open(prep_trigger_data + trigger + '.txt', 'w+')
    input_aux_type = open(prep_trigger_data + aux_type + '.txt', 'w+')
    input_sen = open(prep_trigger_data + sen + '.txt', 'w+')

    for _sent in _sentences:
        # # input vector, List, Size:19
        # _sent.sen_vec
        # # input vector with feature, List, Size:33
        # _sent.sen_vec_feature
        # # tag(1 for positive case, and 0 for negative case), Int, Size: 1
        # _sent.trigger_label
        # # trigger word for the error analysis, Str
        # _sent.trigger
        # # trigger word auxiliary type for the experiment, Str
        # _sent.aux_type
        # # the original sentence for the error analysis, Str
        # _sent.sen

        input_sen_vec.write(str(_sent.sen_vec) + '\n')
        input_sen_vec_feature.write(str(_sent.sen_vec_feature) + '\n')
        input_trigger_label.write(str(_sent.trigger_label) + '\n')
        input_trigger.write(str(_sent.trigger) + '\n')
        input_aux_type.write(str(_sent.aux_type) + '\n')
        input_sen.write(str(_sent.sen) + '\n')

    input_sen_vec.close()
    input_sen.close()
    input_trigger_label.close()
    input_trigger.close()
    input_aux_type.close()
    input_sen.close()
