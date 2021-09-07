

# NVPERC

This is the implement of **A Neural Approach for Verb Phrase Ellipsis Resolution (AAAI-19)**.

# Code

The codes are consist of two parts:

- ```/MLP``` includes the codes and experiments for MLP Classifier, which is contributed by **[LeeeeoLiu](https://github.com/LeeeeoLiu)**.
- ```/T2T``` for Tensor2Tensor Classifier, which is contributed by **[Daviddddl](https://github.com/Daviddddl)**.

> Caution: the MLP is based on PyTorch and Python2, however, the T2T is based on Tensorflow and Python3.

# Data

The data is too large to upload to Github. You can download from the following two ways :

- Google Drive: https://drive.google.com/drive/folders/1DMUkklYA0zY-uXfJEOkFCNMfIAke4m-w?usp=sharing 

- Baidu Cloud Disk: https://pan.baidu.com/s/1AQefeMv9UfPVO_AU7NGx0Q 
  - Extraction Code: jpgv 

You can also use the ```MLP/prepare_data.py``` to generate the training data.

# Citation

If our code is helpful for your research, please cite us.
```
@inproceedings{zhang2019neural,
  title={A neural network approach to verb phrase ellipsis resolution},
  author={Zhang, Wei-Nan and Zhang, Yue and Liu, Yuanxing and Di, Donglin and Liu, Ting},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={7468--7475},
  year={2019}
}
```
