# coding=utf-8


count_lines = []

antecedent_label = 'antecedent_label'
aux_type = 'aux_type'
input_vec_attention = 'input_vec_attention'
input_vec_attention_feature = 'input_vec_attention_feature'
input_vec_hidden = 'input_vec_hidden'
input_vec_hidden_feature = 'input_vec_hidden_feature'
input_vec_sum = 'input_vec_sum'
input_vec_sum_feature = 'input_vec_sum_feature'
sen = 'sen'
trigger = 'trigger'
trigger_label = 'trigger_label'

prep_data = '../prep_ante_data/'
file_list=[
    antecedent_label, aux_type,
    input_vec_attention, input_vec_attention_feature,
    input_vec_hidden, input_vec_hidden_feature,
    input_vec_sum, input_vec_sum_feature,
    sen, trigger, trigger_label
]

for _file in file_list:
    count = 0
    for index, line in enumerate(open(prep_data + _file + '.txt', 'r')):
        count += 1
    count_lines.append(count)
print(count_lines)
