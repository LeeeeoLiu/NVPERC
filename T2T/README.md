# ante_t2t

## Tensorflow1.8 + CUDA9.0 + python3.6 + Tensor2tensor

Run the commands below in terminal:
eg.

```shell
  
  PROBLEM_NAME='sum_pooling'
  DATA_DIR='../train_data_sum'
  OUTPUT_DIR='../output_sum'
  t2t-datagen --t2t_usr_dir=. --data_dir=$DATA_DIR --tmp_dir=../tmp_data --problem=$PROBLEM_NAME
  t2t-trainer --t2t_usr_dir=. --data_dir=$DATA_DIR --problem=$PROBLEM_NAME --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR

  DECODE_FROM = '../decode_in/sum_pooling_in.txt'
  DECODE_TO = '../decode_out_truth/sum_pooling_out.txt'
  t2t-decoder --t2t_usr_dir=./ --problem=$PROBLEM_NAME--data_dir=DATA_DIR --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR --decode_hparams=”beam_size=5,alpha=0.6” --decode_from_file=$DECODE_FROM --decode_to_file=$DECODE_TO
  
```
