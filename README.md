# End-to-end Neural Verb Phrase Ellipsis Resolution

![image-20180816101109172](https://ws4.sinaimg.cn/large/006tNbRwly1fubbjl7u1hj31kw1a5dtk.jpg)

Table of Contents
=================

   * [End-to-end Neural Verb Phrase Ellipsis Resolution](#end-to-end-neural-verb-phrase-ellipsis-resolution)
      * [Requirements](#requirements)
         * [Python Package](#python-package)
         * [Pretrained FastText Model](#pretrained-fasttext-model)
         * [Pretrained OpenNMT Model](#pretrained-opennmt-model)
      * [Data Manager](#data-manager)
      * [Trigger Detection Results](#trigger-detection-results)
      * [Antecedent Indentification Results](#antecedent-indentification-results)

## Requirements

### Python Package

```bash
pip install -r requirements.txt
```
requirements.txt
```bash
fasttext=0.8.3
numpy=1.14.3
nltk=3.2.5
Pattern=2.6
scikit-learn=0.19.1
scipy=0.19.0
torch=0.4.0
```

### Pretrained FastText Model

This model is used to get the word embedding in the antecedent identification part.

### Pretrained OpenNMT Model

This model is used to get the words hidden and attention representations in the antecedent identification part.

**You can download the two pretrained models from [here](https://pan.baidu.com/s/1_i5GkgvNL0-rN0jvyvuR8Q) **

Then put `fasttext_model_4_all_wsj.bin` into the `datas/` , put `encoder_model.pt` into the `datas/data_4_OpenNMT/`

## Data Manager

Data Manager provided a easy way to get the infomation of every sentence. The usages are showed in the following example code.

```python
# coding=utf-8

import argparse
from DataManager import DataManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameters description")
    parser.add_argument('-train_test', type=bool, default=False,
                        help="Type:bool. Whether use train-test proposed by Bos \nDefalut:False")
    parser.add_argument('-feature', type=bool, default=False,
                        help="Type:bool. Whether add feature to the SVM Classification. \nDefalut:False")
    parser.add_argument('-overwrite', type=bool, default=False,
                        help="Type:bool. Whether restart processing all the data. \nDefalut:False")
    args = parser.parse_args()
    # vars make args to dict, for example: {'train_test':False, 'feature':False, 'overwrite':False}
    dm = DataManager(vars(args))
    dm.init()
    
    # get all the sentences for trigger detection
    _sentences = dm.trigger_generate_sentences	# list
    
    for _sent in _sentences:
    	# input vector, List, Size:19
    	_sent.sen_vec
    	# input vector with feature, List, Size:33
    	_sent.sen_vec_feature
    	# tag(1 for positive case, and 0 for negative case), Int, Size: 1
    	_sent.trigger_label
    
    # get all the sentences for antecedent identification
    _sentences =  dm.antecedent_generate_sentences	# list
    
    for _sent in _sentences:
    	# sum pooling, FloatTensor, Size: 400
        _sent.input_vec_sum
        # sum pooling with feature, FloatTensor, Size: 468
        _sent.input_vec_sum_feature
        # GRU, FloatTensor, Size: 6100
        _sent.input_vec_hidden
        # GRU with feature, FloatTensor, Size: 6168
        _sent.input_vec_hidden_feature
        # AttentionGRU, FloatTensor, Size: 1600
        _sent.input_vec_attention
        # AttentionGRU with feature, FloatTensor, Size: 1668
        _sent.input_vec_attention_feature
        # tag(1 for positive case, and 0 for negative case), Int, Size: 1
        _sent.antecedent_label
```

Besides, you can get more info in a sentence object. The defination of the Sentence Class is as follows.
```python
class Sentence(object):
    """ A Sentence Data Class. """
    def __init__(self, sen_tagged, wsj_section):
        """ basic info """
        # sen_tagged with BIOEST, for example: Not/RB/O only/RB/O ... .
        self.sen_tagged = sen_tagged
        # the wsj section the sentence come from, for example: wsj.section00.
        self.wsj_section = wsj_section
        # the sentence, for example: Not only ... .
        self.sen = None 
        # show whether the sentence is gold or generated
        self.is_generated = False
        # a list of words in the sentence, for example: ['Not', 'only', ...].
        self.words_list = []
        # a list of POS index(refer to POS_2_INDEX below) of the words in the sentence, for example: [19, 19, ...].
        self.pos_list = []
        # a list of tag index(refer to TAG_2_INDEX below) of the words in the sentence, for example: [2, 2, ...].
        self.tag_list = []

        """ for trigger training """
        # the trigger word in the sentence, for example: is.
        self.sen_trigger_word = None
        # the auxiliay type of the trigger, for example: be.
        self.sen_trigger_aux_type = None
        # the training input vector
        self.sen_vec = None
        # the training input vector with feature
        self.sen_vec_feature = None
        # tag (1 for positive case, 0 for negative case), for example: 1
        self.trigger_label = None

        """ for antecedent training """
        # the berkely parse result of the sentence
        self.sen_parse = None
        # the trigger index in the words list
        self.trigger_index = None
        # list which marked the gold antecedent with 1, otherwise 0
        self.truth_label = None
        # words before antecedent
        self.words_before_antecedent = None
        # antecedent words
        self.antecedent = None
        # words after antecedent
        self.words_after_antecedent = None
        # list which marked the words with 0(other), 1(antecedent) and 2(trigger) 
        self.sen_tag_label = None
        
        # hidden
        # GRU hidden, Tensor, Size: sentences_length X 6000
        self.before_antecedent_hidden_tensor = None
        self.antecedent_hidden_tensor = None
        self.after_antecedent_hidden_tensor = None

        # attention
        # GRU hidden, Tensor, Size: sentences_length X 1500
        self.before_antecedent_attention_tensor = None
        self.antecedent_attention_tensor = None
        self.after_antecedent_attention_tensor = None

        # input vec 
        # sum pooling, FloatTensor, Size: 400
        self.input_vec_sum = None
        # sum pooling with feature, FloatTensor, Size: 468
        self.input_vec_sum_feature = None
        # GRU, FloatTensor, Size: 6100
        self.input_vec_hidden = None
        # GRU with feature, FloatTensor, Size: 6168
        self.input_vec_hidden_feature = None
        # AttentionGRU, FloatTensor, Size: 1600
        self.input_vec_attention = None
        # AttentionGRU with feature, FloatTensor, Size: 1668
        self.input_vec_attention_feature = None
        # tag(1 for positive case, and 0 for negative case), Int, Size: 1
        self.antecedent_label = None
```

## Trigger Detection Results

| Auxiliary  | ML | SVM | SVM+Feature |
| :-------: | :----: | :-----: | :--: |
| Do  | 0.89 | 0.90 |      |
| Be  | 0.63 | 0.63 |      |
| Have  | 0.75 | 0.61 |      |
| Modal | 0.86 | 0.72 |      |
| To | 0.79 | 0.44 |      |
| So | 0.86 | 0.90 |      |
| ALL | 0.82 | 0.81 |      |

**Table 1**: VPE detection results (baseline F1, Machine Learning F1, SVM F1, SVM with Auxiliary and Syntactic F1) obtained with 5-fold cross validation. (ML refers to the EMNLP 2016 paper of  Kenyon-Dean K et al.)

| Test Set Results           |   P    |   R    |   F1   |
| :------------------------- | :----: | :----: | :----: |
| Liu et al.(2016)           | 0.8022 | 0.6135 | 0.6953 |
| Kenyon-Dean K et al.(2016) | 0.7574 | 0.8655 | 0.8078 |
| SVM                        | 0.8089 | 0.7983 | 0.7966 |
| SVM+Feature                | 0.9541 | 0.9538 | 0.9538 |

**Table 2**: Results (precision, recall, F1) for VPE detection using the train-test split proposed by Bus and Spenader(2011) 

## Antecedent Indentification Results

| Auxiliary | Baseline | MIRA |  SUM MLP    |   SUM MLP Feature   |  RNN MLP    | RNN MLP Feature     |  RNN Attention MLP    | RNN Attention MLP Feature     |
| :-----------: | :----------: | :------: | :--: | :--: | :--: | :--: | :--: | :--: |
| do | 0.42 | 0.71 |      |      |      |      |      |      |
| be | 0.37 | 0.63 |      |      |      |      |      |      |
| modal | 0.42 | 0.67 |      |      |      |      |      |      |
| so | 0.15 | 0.53 |      |      |      |      |      |      |
| have | 0.39 | 0.61 |      |      |      |      |      |      |
| to | 0.03 | 0.58 |      |      |      |      |      |      |
| ALL | 0.36 | 0.65 |      |      |      |      |      |      |

**Table 3**: Results（baseline accuracy, MIRA accuracy, Sum-Pooling MLP accuracy, Sum-Pooling MLP with Feature accuracy, RNN-Encoder MLP accuracy, RNN-Encoder MLP with Feature accuracy, RNN-Attention-Encoder MLP accuracy, RNN-Attention-Encoder MLP with Feature accuracy）for antecedent identification; obtained with 5-fold cross validation. 

| End-to-end  Results                     |   P    |   R    |   F1   |
| --------------------------------------- | :----: | :----: | :----: |
| Liu et al.(2016)                        | 0.5482 | 0.4192 | 0.4751 |
| Kenyon-Dean K et al.(2016)              | 0.4871 | 0.5567 | 0.5196 |
| Sum pooling + MLP                       |        |        |        |
| Sum pooling + MLP + Feature             |        |        |        |
| RNN-Encoder + MLP                       |        |        |        |
| RNN-Encoder + MLP + Feature             |        |        |        |
| RNN-Encoder + Attention + MLP           |        |        |        |
| RNN-Encoder + Attention + MLP + Feature |        |        |        |

**Table 4**: End-to-end results (precision, recall, F1) using the train-test split proposed by Bos and Spenader (2011). 

