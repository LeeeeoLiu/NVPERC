# coding=utf-8

import argparse
from datetime import datetime
from DataManager import DataManager

if __name__ == '__main__':
    start = datetime.now()
    parser = argparse.ArgumentParser(description="Parameters description")
    parser.add_argument('-train_test', type=bool, default=False, help="Type:bool. Whether use train-test proposed by Bos \nDefalut:False")
    parser.add_argument('-feature', type=bool, default=False, help="Type:bool. Whether add feature to the SVM Classification. \nDefalut:False")
    parser.add_argument('-overwrite', type=bool, default=False, help="Type:bool. Whether restart processing all the data. \nDefalut:False")
    args = parser.parse_args()
   
    dm = DataManager(vars(args))
    dm.init()
    end = datetime.now()
    print 'Cost Time: {}'.format(end-start)
