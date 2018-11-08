# coding=utf-8
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem, text_problems
import pickle as pickle

@registry.register_problem
class AttentionGruFeature(text_problems.Text2ClassProblem):

    ROOT_DATA_PATH = '../data_manager/'
    PROBLEM_NAME = 'attention_gru_feature'

    @property
    def is_generate_per_split(self):
        return True

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 5,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def approx_vocab_size(self):
        return 2 ** 10  # 8k vocab suffices for this small dataset.

    @property
    def num_classes(self):
        return 2

    @property
    def vocab_filename(self):
        return self.PROBLEM_NAME + ".vocab.%d" % self.approx_vocab_size

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        # with open('{}self_antecedent_generate_sentences.pkl'.format(self.ROOT_DATA_PATH), 'rb') as f:
        #     # get all the sentences for antecedent identification
        #     _sentences = pickle.load(f)
        #
        # for _sent in _sentences:
        #     # # sum pooling, FloatTensor, Size: 400
        #     # _sent.input_vec_sum
        #     # # sum pooling with feature, FloatTensor, Size: 468
        #     # _sent.input_vec_sum_feature
        #     # # GRU, FloatTensor, Size: 6100
        #     # _sent.input_vec_hidden
        #     # # GRU with feature, FloatTensor, Size: 6168
        #     # _sent.input_vec_hidden_feature
        #     # # AttentionGRU, FloatTensor, Size: 1600
        #     # _sent.input_vec_attention
        #     # # AttentionGRU with feature, FloatTensor, Size: 1668
        #     # _sent.input_vec_attention_feature
        #     # # tag(1 for positive case, and 0 for negative case), Int, Size: 1
        #     # _sent.antecedent_label
        #     # # tag(1 for positive case, and 0 for negative case), Int, Size: 1
        #     # _sent.trigger_label
        #     # # trigger word for the error analysis, Str
        #     # _sent.trigger
        #     # # trigger word auxiliary type for the experiment, Str
        #     # _sent.aux_type
        #     # # the original sentence for the error analysis, Str
        #     # _sent.sen
        #
        #     yield {
        #         "inputs": _sent.input_vec_attention_feature,
        #         "label": _sent.antecedent_label
        #     }

        with open('../prep_ante_data/antecedent_label.txt') as antecedent_label, open(
                '../prep_ante_data/input_vec_attention_gru_feature.txt') as input_vec:
            for labal in antecedent_label:
                yield {
                    "inputs": input_vec.readline().strip()[1:-2],
                    "label": int(labal.strip())
                }

        antecedent_label.close()
        input_vec.close()


# PROBLEM_NAME='attention_gru_feature'
# DATA_DIR='../train_data_atte_feature'
# OUTPUT_DIR='../output_atte_feature'
# t2t-datagen --t2t_usr_dir=. --data_dir=$DATA_DIR --tmp_dir=../tmp_data --problem=$PROBLEM_NAME
# t2t-trainer --t2t_usr_dir=. --data_dir=$DATA_DIR --problem=$PROBLEM_NAME --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR
