# End-to-end Neural Verb Phrase Ellipsis Resolution

![image-20180823134307677](https://ws2.sinaimg.cn/large/006tNbRwly1fujl0e7wqoj31kw10p13j.jpg)

Table of Contents
=================

   * [End-to-end Neural Verb Phrase Ellipsis Resolution](#end-to-end-neural-verb-phrase-ellipsis-resolution)
   * [Table of Contents](#table-of-contents)
      * [Requirements](#requirements)
         * [Python Package](#python-package)
         * [Pretrained FastText Model](#pretrained-fasttext-model)
         * [Pretrained OpenNMT Model](#pretrained-opennmt-model)
      * [Data Manager](#data-manager)
         * [Step 1 Pretrain Data Manager](#step-1-pretrain-data-manager)
         * [Step 2 Use Pretrained Data](#step-2-use-pretrained-data)
      * [Experiments](#experiments)
         * [Trigger Detection Results (VPE Detection)](#trigger-detection-results-vpe-detection)
         * [Antecedent Indentification Results (VPE Resolution)](#antecedent-indentification-results-vpe-resolution)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)

## Requirements

### Python Package

```bash
pip install -r requirements.txt
```

### Pretrained FastText Model

This model is used to get the word embedding in the antecedent identification part.

### Pretrained OpenNMT Model

This model is used to get the words hidden and attention representations in the antecedent identification part.

**You can download the two pretrained models from** [here](https://pan.baidu.com/s/1_i5GkgvNL0-rN0jvyvuR8Q)

Then put `fasttext_model_4_all_wsj.bin` into the `datas/` , put `encoder_model.pt` into the `datas/data_4_OpenNMT/`

## Data Manager

Data Manager provided a easy way to get the infomation of every sentence. The usages are showed in the following example code.

### Step 1 Pretrain Data Manager
Run the following code to pretrain the Data Manager. If you want to re-train Data Manager,  just add the parameter `-overwrite True`.

```bash
python prepare.py
```

### Step 2 Use Pretrained Data

For Trigger Detection,  you can get the datas by the following way:
```python
# coding=utf-8

if __name__ == '__main__':
	ROOT_DATA_PATH = './datas/data_manager/'
    with open('{}self_trigger_generate_sentences.pkl'.format(ROOT_DATA_PATH),'r') as f:
        # get all the sentences for trigger detection
        _sentences = pickle.load(f)
    
    for _sent in _sentences:
    	# input vector, List, Size:19
    	_sent.sen_vec
    	# input vector with feature, List, Size:33
    	_sent.sen_vec_feature
    	# tag(1 for positive case, and 0 for negative case), Int, Size: 1
    	_sent.trigger_label
        # trigger word for the error analysis, Str
        _sent.trigger
        # trigger word auxiliary type for the experiment, Str
        _sent.aux_type
        # the original sentence for the error analysis, Str
        _sent.sen
```

For Antecedent Identification,  you can get the datas by the following way:
```python
# coding=utf-8

if __name__ == '__main__':
	ROOT_DATA_PATH = './datas/data_manager/'
    with open('{}self_antecedent_generate_sentences.pkl'.format(ROOT_DATA_PATH),'r') as f:
        # get all the sentences for antecedent identification
        _sentences = pickle.load(f)
    
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
    	# tag(1 for positive case, and 0 for negative case), Int, Size: 1
    	_sent.trigger_label
        # trigger word for the error analysis, Str
        _sent.trigger
        # trigger word auxiliary type for the experiment, Str
        _sent.aux_type
        # the original sentence for the error analysis, Str
        _sent.sen
```

## Experiments

The experiments are consist of VPE Detection and VPE Resolution. 
### Trigger Detection Results (VPE Detection)

**Please refer to `VPE_Detection.py`**

```bash
usage: VPE_Detection_SVM.py [-h] [-train_test TRAIN_TEST] [-aux AUX]

optional arguments:
  -h, --help            show this help message and exit
  -train_test TRAIN_TEST
                        Type:bool. Whether use train-test proposed by Bos for
                        training. Defalut:False
  -aux AUX              Type:bool. Whether show classification report for each
                        Auxiliary. Defalut:False
```



| Auxiliary  | ML | SVM | SVM+Feature |
| :-------: | :----: | :-----: | :--: |
| Do  | 0.89 | **0.94** | 0.93 |
| Be  | 0.63 | 0.71 | **0.76** |
| Have  | 0.75 | 0.76 | **0.90** |
| Modal | 0.86 | **0.95** | **0.95** |
| To | 0.79 | 0.64 | **0.86** |
| So | 0.86 | **0.91** | 0.90 |
| ALL | 0.82 | 0.87 | **0.90** |

**Table 1**: VPE detection results (baseline F1, Machine Learning F1, SVM F1, SVM with Auxiliary and Syntactic F1) obtained with 5-fold cross validation. (ML refers to the EMNLP 2016 paper of  Kenyon-Dean K et al.)

| Test Set Results           |     P      |     R      |     F1     |
| :------------------------- | :--------: | :--------: | :--------: |
| Liu et al.(2016)           |   0.8022   |   0.6135   |   0.6953   |
| Kenyon-Dean K et al.(2016) |   0.7574   |   0.8655   |   0.8078   |
| SVM                        |   0.8803   |   0.8782   |   0.8780   |
| SVM+Feature                | **0.9048** | **0.9034** | **0.9033** |

**Table 2**: Results (precision, recall, F1) for VPE detection using the train-test split proposed by Bus and Spenader(2011) 

### Antecedent Indentification Results (VPE Resolution)
**Please refer to `VPE_Resolution.py`**
```bash
usage: VPE_Resolution_MLP.py [-h] [-train_test TRAIN_TEST] [-aux AUX]
                             [-gpu GPU] [-model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -gpu GPU              Use GPU Id Defalut:-1
  -model MODEL          Choose Model Defalut:0(SUM)
```

And the model list is as follows:
- 0, SUM Pooling + MLP
- 1, SUM Pooling + MLP + Feature
- 2, GRU + MLP
- 3, GRU + MLP + Feature
- 4, GRU + Attention + MLP
- 5, GRU + Attention + MLP + Feature

For example, `python VPE_Resolution_MLP.py -gpu 1 -model 5`.


| Auxiliary | Baseline | MIRA |  SUM MLP    |   SUM MLP Feature   |  GRU MLP  | GRU MLP Feature  |  GRU Attention MLP  | GRU Attention MLP Feature  |
| :-----------: | :----------: | :------: | :--: | :--: | :--: | :--: | :--: | :--: |
| do | 0.42 | 0.71 | 0.84 | 0.91 | 0.81 | 0.90 | 0.86 | **0.93** |
| be | 0.37 | 0.63 | 0.86 | 0.89 | 0.78 | 0.84 | 0.85 | **0.96** |
| modal | 0.42 | 0.67 | 0.86 | 0.90 | 0.84 | 0.89 | 0.78 | **0.94** |
| so | 0.15 | 0.53 | 0.81 | 0.88 | 0.79 | 0.79 | 0.85 | **0.90** |
| have | 0.39 | 0.61 | 0.79 | 0.79 | 0.76 | 0.87 | 0.76 | **0.89** |
| to | 0.03 | 0.58 | 0.84 | **0.97** | 0.87 | 0.78 | 0.87 | 0.93 |
| ALL | 0.36 | 0.65 | 0.80 | 0.90 | 0.80 | 0.86 | 0.83 | **0.93** |

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

