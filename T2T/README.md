# ante_t2t

## Tensorflow1.8 + CUDA9.0 + python3.6 + Tensor2tensor

#### You can create an environment of anaconda

Step 1: download and install Anaconda 
  https://www.anaconda.com/download/#linux
  for more install details: https://blog.csdn.net/Davidddl/article/details/81873606
```
sudo bash Anaconda3-5.2.0-Linux-x86_64.sh

conda --version

conda create -n tensorflow pip python=3.6

source activate tensorflow

pip install --upgrade pip

(tensorflow)$ pip install --ignore-installed --upgrade https://download.tensorflow.google.cn/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
```

Step 2: install tensor2tensor
```
(tensorflow)$ pip install tensor2tensor
```

Step 3: define some parameters
```
  PROBLEM_NAME='sum_pooling'   
  # this is your problem name, maybe is attention_gru, attention_gru_feature, etc..
  # Every problem's name is writen in every script's comments
  DATA_DIR='../train_data_sum'
  OUTPUT_DIR='../output_sum'
```

Step 4: generate t2t date
```
t2t-datagen --t2t_usr_dir=. --data_dir=$DATA_DIR --tmp_dir=../tmp_data --problem=$PROBLEM_NAME
```
Step 5: train your model
```
 t2t-trainer --t2t_usr_dir=. --data_dir=$DATA_DIR --problem=$PROBLEM_NAME --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR
```
Step 6: decode
```
  DECODE_FROM = '../decode_in/sum_pooling_in.txt'
  DECODE_TO = '../decode_out_truth/sum_pooling_out.txt'
  t2t-decoder --t2t_usr_dir=./ --problem=$PROBLEM_NAME--data_dir=DATA_DIR --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR --decode_hparams=”beam_size=5,alpha=0.6” --decode_from_file=$DECODE_FROM --decode_to_file=$DECODE_TO
```
