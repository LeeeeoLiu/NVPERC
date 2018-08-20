# coding=utf-8

import argparse
from DataManager import DataManager

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Parameters description")
	parser.add_argument('-train_test', type=bool, default=False, help="Type:bool. Whether use train-test proposed by Bos \nDefalut:False")
	parser.add_argument('-feature', type=bool, default=False, help="Type:bool. Whether add feature to the SVM Classification. \nDefalut:False")
	parser.add_argument('-overwrite', type=bool, default=False, help="Type:bool. Whether restart processing all the data. \nDefalut:False")
	args = parser.parse_args()
	dm = DataManager(vars(args))
	dm.init()
	_sentences =  dm.antecedent_generate_sentences	# list
	for _sent in _sentences:
		print _sent.input_vec_sum
		print _sent.input_vec_sum_feature
		print _sent.input_vec_hidden
		print _sent.input_vec_hidden_feature
		print _sent.input_vec_attention
		print _sent.input_vec_attention_feature
		print _sent.antecedent_label

