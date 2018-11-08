# coding=utf-8
import os,sys

def main(name, total, flag):
    path_data_from_trans = "./"
    path_data_from_truth = "../decode_out_truth"
    files_from_trans = os.listdir(path_data_from_trans)
    files_from_truth = os.listdir(path_data_from_truth)

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    task_name = name

    decode_out_truth = '../decode_out_truth'
    decode_out_from_trans = '../decode_out_from_trans/'
    parent_task_name = 'Wsj_out_from_trans_'
    task_file_name = parent_task_name + task_name + '_'
    decode_out_from_trans_count = '../decode_out_from_trans_count/'

    Auxiliary_Do = ['do']
    Auxiliary_Be = ['be']
    Auxiliary_So = ['so', 'same', 'likewise', 'opposite']
    Auxiliary_Have = ['have']
    Auxiliary_To = ['to']
    Auxiliary_Modal = ['will', 'would', 'can', 'could', 'should', 'may', 'might', 'must']

    all_sum = 0
    do_sum = 0
    do_correct = 0
    be_sum = 0
    be_correct = 0
    so_sum = 0
    so_correct = 0
    have_sum = 0
    have_correct = 0
    to_sum = 0
    to_correct = 0
    modal_sum = 0
    modal_correct = 0

    f = open(decode_out_from_trans_count + task_name + '.txt', 'w+')

    for i in range(20, 25):
        with open(decode_out_truth + '/Wsj.Section' + str(i) + '_out.txt') as truth, open(
                decode_out_from_trans + task_file_name + str(i) + '.txt') as trans:

            for l1 in truth:
                l1 = l1.strip()
                l2 = trans.readline().strip()

                f.write('Truth=' + l1 + ' === Predict='+l2[-1] + '\n')
    f.close()


    with open(decode_out_from_trans_count + task_name + '.txt') as res:
        for line in res:

            all_sum += 1

            if 'Truth=1' in line and 'Predict=1' in line:
                TP += 1
            if 'Truth=1' in line and 'Predict=0' in line:
                FN += 1
            if 'Truth=0' in line and 'Predict=1' in line:
                FP += 1
            if 'Truth=0' in line and 'Predict=0' in line:
                TN += 1

            if 'do' in line:
                do_sum += 1
                if 'Truth=1' in line and 'Predict=1' in line:
                    do_correct += 1
                if 'Truth=0' in line and 'Predict=0' in line:
                    do_correct += 1

            if 'be' in line:
                be_sum += 1
                if 'Truth=1' in line and 'Predict=1' in line:
                    be_correct += 1
                if 'Truth=0' in line and 'Predict=0' in line:
                    be_correct += 1

            for so in Auxiliary_So:
                if so in line:
                    so_sum += 1
                    if 'Truth=1' in line and 'Predict=1' in line:
                        so_correct += 1
                    if 'Truth=0' in line and 'Predict=0' in line:
                        so_correct += 1

            if 'have' in line:
                have_sum += 1
                if 'Truth=1' in line and 'Predict=1' in line:
                    have_correct += 1
                if 'Truth=0' in line and 'Predict=0' in line:
                    have_correct += 1

            if 'to' in line:
                to_sum += 1
                if 'Truth=1' in line and 'Predict=1' in line:
                    to_correct += 1
                if 'Truth=0' in line and 'Predict=0' in line:
                    to_correct += 1

            for modal in Auxiliary_Modal:
                if modal in line:
                    modal_sum += 1
                    if 'Truth=1' in line and 'Predict=1' in line:
                        modal_correct += 1
                    if 'Truth=0' in line and 'Predict=0' in line:
                        modal_correct += 1


    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    print('do = ' + str(do_correct/do_sum))
    print('be = ' + str(be_correct/be_sum))
    print('have = ' + str(have_correct/have_sum))
    print('modal = ' + str(modal_correct/modal_sum))
    print('to = ' + str(to_correct/to_sum))
    print('so = ' + str(so_correct/so_sum))
    print('all = ' + str((TP+TN)/all_sum))

    print('')
    print('P = ' + str(P))
    print('R = ' + str(R))
    print('F1 = ' + str(F1))

    print ('')
    print('FP = ' + str(FP))
    print('FN = ' + str(FN))
    print('TP = ' + str(TP))
    print('TN = ' + str(TN))
