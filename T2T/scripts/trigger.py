# coding=utf-8
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem, text_problems
import pickle as pickle

@registry.register_problem
class Trigger(text_problems.Text2ClassProblem):

    ROOT_DATA_PATH = '../data_manager/'
    PROBLEM_NAME = 'trigger'

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

        # with open('{}self_trigger_generate_sentences.pkl'.format(ROOT_DATA_PATH), 'rb') as f:
        #     # get all the sentences for trigger detection
        #     _sentences = pickle.load(f)

        # for _sent in _sentences:
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
        #
        #     yield {
        #         "inputs": _sent.sen_vec,
        #         "label": _sent.trigger_label
        #     }

        with open('../prep_trigger_data/trigger_label.txt') as trigger_label, open(
                '../prep_trigger_data/sen_vec.txt') as input_vec:
            for labal in trigger_label:
                yield {
                    "inputs": input_vec.readline().strip()[1:-2],
                    "label": int(labal.strip())
                }

        trigger_label.close()
        input_vec.close()


# PROBLEM_NAME='sum_pooling'
# DATA_DIR='../train_data_sum'
# OUTPUT_DIR='../output_sum'
# t2t-datagen --t2t_usr_dir=. --data_dir=$DATA_DIR --tmp_dir=../tmp_data --problem=$PROBLEM_NAME
# t2t-trainer --t2t_usr_dir=. --data_dir=$DATA_DIR --problem=$PROBLEM_NAME --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR

# DECODE_FROM = '../decode_in/sum_pooling_in.txt'
# DECODE_TO = '../decode_out_truth/sum_pooling_out.txt'
# t2t-decoder --t2t_usr_dir=./ --problem=$PROBLEM_NAME--data_dir=DATA_DIR --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR --decode_hparams=”beam_size=5,alpha=0.6” --decode_from_file=$DECODE_FROM --decode_to_file=$DECODE_TO
