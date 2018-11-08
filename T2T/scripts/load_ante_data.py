# coding=utf-8
import pickle as pickle

if __name__ == '__main__':
    ROOT_DATA_PATH = '../data_manager/'
    with open('{}self_antecedent_generate_sentences.pkl'.format(ROOT_DATA_PATH), 'rb') as f:
        # get all the sentences for antecedent identification
        _sentences = pickle.load(f)

    prep_ante_data = '../prep_ante_data/'
    input_vec_sum = open(prep_ante_data + "input_vec_sum.txt", 'w+')
    input_vec_sum_feature = open(prep_ante_data + "input_vec_sum_feature.txt", 'w+')
    input_vec_hidden = open(prep_ante_data + "input_vec_hidden.txt", 'w+')
    input_vec_hidden_feature = open(prep_ante_data + "input_vec_hidden_feature.txt", 'w+')
    input_vec_attention = open(prep_ante_data + "input_vec_attention.txt", 'w+')
    input_vec_attention_feature = open(prep_ante_data + "input_vec_attention_feature.txt", 'w+')
    antecedent_label = open(prep_ante_data + "antecedent_label.txt", 'w+')
    trigger_label = open(prep_ante_data + "trigger_label.txt", 'w+')
    trigger = open(prep_ante_data + "trigger.txt", 'w+')
    aux_type = open(prep_ante_data + "aux_type.txt", 'w+')
    sen = open(prep_ante_data + "sen.txt", 'w+')

    for _sent in _sentences:
        # # sum pooling, FloatTensor, Size: 400
        # _sent.input_vec_sum
        # # sum pooling with feature, FloatTensor, Size: 468
        # _sent.input_vec_sum_feature
        # # GRU, FloatTensor, Size: 6100
        # _sent.input_vec_hidden
        # # GRU with feature, FloatTensor, Size: 6168
        # _sent.input_vec_hidden_feature
        # # AttentionGRU, FloatTensor, Size: 1600
        # _sent.input_vec_attention
        # # AttentionGRU with feature, FloatTensor, Size: 1668
        # _sent.input_vec_attention_feature
        # # tag(1 for positive case, and 0 for negative case), Int, Size: 1
        # _sent.antecedent_label
        # # tag(1 for positive case, and 0 for negative case), Int, Size: 1
        # _sent.trigger_label
        # # trigger word for the error analysis, Str
        # _sent.trigger
        # # trigger word auxiliary type for the experiment, Str
        # _sent.aux_type
        # # the original sentence for the error analysis, Str
        # _sent.sen

        input_vec_sum.write(str(_sent.input_vec_sum) + '\n')
        input_vec_sum_feature.write(str(_sent.input_vec_sum_feature) + '\n')
        input_vec_hidden.write(str(_sent.input_vec_hidden) + '\n')
        input_vec_hidden_feature.write(str(_sent.input_vec_hidden_feature) + '\n')
        input_vec_attention.write(str(_sent.input_vec_attention) + '\n')
        input_vec_attention_feature.write(str(_sent.input_vec_attention_feature) + '\n')
        antecedent_label.write(str(_sent.antecedent_label) + '\n')
        trigger_label.write(str(_sent.trigger_label) + '\n')
        trigger.write(str(_sent.trigger) + '\n')
        aux_type.write(str(_sent.aux_type) + '\n')
        sen.write(str(_sent.sen) + '\n')

    input_vec_sum.close()
    input_vec_sum_feature.close()
    input_vec_hidden.close()
    input_vec_hidden_feature.close()
    input_vec_attention.close()
    antecedent_label.close()
    input_vec_attention_feature.close()
    trigger_label.close()
    trigger.close()
    aux_type.close()
    sen.close()
