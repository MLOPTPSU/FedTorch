# -*- coding: utf-8 -*-
import re
import pandas as pd
import os
import numpy as np
import torch

from fedtorch.utils.auxiliary import str2time, is_float
from fedtorch.utils.op_files import read_txt

ZERO_TIME_TRAIN=None
ZERO_TIME_TEST=None
VAR_NAMES_TRAIN = [
        'time', 'epoch', 'local_index',
        'load_time', 'data_time', 'compute_time', 'sync_time',
        'global_time', 'loss', 'top1', 'top5','learning_rate']
# VAR_NAMES_TEST = ['time', 'info', 'best_epoch', 'current_epoch', 'top1']
VAR_NAMES_TEST = ['time', 'batch', 'epoch', 'Process', 'top1', 'top5', 'loss','comm']
VAR_NAMES_VAL = ['time', 'batch', 'epoch', 'Process', 'top1', 'top5','loss','comm']
VAR_NAMES_STAT = ['time', 'batch', 'epoch', 'Process', 'worst', 'best','var','comm']
VAR_NAMES_WEIGHT = ['time','distance']
VAR_NAMES_ALPHA =  ['time','alpha']
VAR_NAMES_COMM =  ['time','comm_time']
VAR_NAMES_PER_CLASS = ['time', 'batch', 'epoch', 'Process', 'acc','comm']
VAR_NAMES_PER_NODE = ['time', 'comm', 'acc']

def _parse_record(lines, parse_fn, pattern, var_names):
    parsed_lines = []
    for line in lines:
        parsed_line = parse_fn(line, pattern, var_names)
        if parsed_line is not None:
            parsed_lines.append(parsed_line)

    return parsed_lines


def _parse_record_for_train_fn(line, pattern, var_names):
    global ZERO_TIME_TRAIN
    try:
        # print(line)
        matched_line = re.findall(pattern, line, re.DOTALL)

        if len(matched_line) != 0:
            # get parsed line.
            matched_line = [x.strip() for x in matched_line[0]]
            # convert the string to time.
            if ZERO_TIME_TRAIN:
                matched_line[0] = (str2time(matched_line[0], '%Y:%m:%d %H:%M:%S') - ZERO_TIME_TRAIN).total_seconds()
            else:
                ZERO_TIME_TRAIN = str2time(matched_line[0], '%Y:%m:%d %H:%M:%S')
                matched_line[0] = 0.0

            # map str to float
            matched_line = [
                float(x) if isinstance(x, str) and is_float(x) else x
                for x in matched_line]
            # build dictionary
            # zip_line = zip(var_names, matched_line)
            # line = dict(zip_line)
            return matched_line
            # return line
        else:
            return None
    except Exception as e:
        print(' the error: {}'.format(e))
        return None


def _parse_record_for_train(lines):
    # pattern = r'(.*?)\s+Process .+: Epoch: (.*?)\. Local index: (.*?)\. Load: (.*?)s \| Data: (.*?)s \| Computing: (.*?)s \| Sync: (.*?)s \| Global: (.*?)s \| Loss: (.*?) \| top1: (.*?) \| top5: (.*?)$'
    pattern = r'(.*?)\s+Process .+: Epoch: (.*?)\. Local index: (.*?)\. Load: (.*?)s \| Data: (.*?)s \| Computing: (.*?)s \| Sync: (.*?)s \| Global: (.*?)s \| Loss: (.*?) \| top1: (.*?) \| top5: (.*?) \| learning_rate: (.*?)$'
    return _parse_record(lines, _parse_record_for_train_fn, pattern, VAR_NAMES_TRAIN)


def _parse_record_for_test_fn(line, pattern, var_names):
    global ZERO_TIME_TEST
    try:
        matched_line = re.findall(pattern, line, re.DOTALL)

        if len(matched_line) != 0:
            # get parsed line.
            matched_line = [x.strip() for x in matched_line[0]]
            # convert the string to time.
            if ZERO_TIME_TEST:
                matched_line[0] = (str2time(matched_line[0], '%Y:%m:%d %H:%M:%S') - ZERO_TIME_TEST).total_seconds()
            else:
                ZERO_TIME_TEST = str2time(matched_line[0], '%Y:%m:%d %H:%M:%S')
                matched_line[0] = 0.0
            # map str to float
            matched_line = [
                float(x) if isinstance(x, str) and is_float(x) else x
                for x in matched_line]
            # build dictionary
            # zip_line = zip(var_names, matched_line)
            # line = dict(zip_line)
            return matched_line
        else:
            return None
    except Exception as e:
        print(' the error: {}'.format(e))
        return None


def _parse_record_for_val_fn(line, pattern, var_names):
    global ZERO_TIME_TEST
    try:
        matched_line = re.findall(pattern, line, re.DOTALL)

        if len(matched_line) != 0:
            # get parsed line.
            matched_line = [x.strip() for x in matched_line[0]]
            # convert the string to time.
            if ZERO_TIME_TEST:
                matched_line[0] = (str2time(matched_line[0], '%Y:%m:%d %H:%M:%S') - ZERO_TIME_TEST).total_seconds()
            else:
                ZERO_TIME_TEST = str2time(matched_line[0], '%Y:%m:%d %H:%M:%S')
                matched_line[0] = 0.0
            # map str to float
            matched_line = [
                float(x) if isinstance(x, str) and is_float(x) else x
                for x in matched_line]
            # build dictionary
            # zip_line = zip(var_names, matched_line)
            # line = dict(zip_line)
            return matched_line
        else:
            return None
    except Exception as e:
        print(' the error: {}'.format(e))
        return None


def _parse_record_for_test(lines):
    # pattern = r'(.*?)\t(.*?)\(best epoch (.*?), current epoch (.*?)\):(.*?)\.$'
    # pattern = r'(.*?)\tTest at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?)$'
    pattern = r'(.*?)\tTest at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?) Loss: (.*?) Comm: (.*?)$'
    

    # lines = [line for line in lines if 'best epoch' in line]
    lines = [line for line in lines if 'Test at batch' in line]
    return _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_TEST)

def _parse_record_for_test_stat(lines):
    pattern = r'(.*?)\tTest per client stat at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Worst: (.*?) Best: (.*?) Var: (.*?) Comm: (.*?)$'
    lines = [line for line in lines if 'Test per client stat at batch' in line]
    return _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_STAT)

def _parse_record_for_test_per_class(lines):
    # pattern = r'(.*?)\t(.*?)\(best epoch (.*?), current epoch (.*?)\):(.*?)\.$'
    # pattern = r'(.*?)\tTest at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?)$'
    pattern = r'(.*?)\tTest per class at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: \[(.*?)\] Comm: (.*?)$'
    

    # lines = [line for line in lines if 'best epoch' in line]
    lines = [line for line in lines if 'Test per class at batch' in line]
    data = _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_PER_CLASS)
    result = [[float(x) for x in row[4].split(', ')] for row in data]
    return result

def _parse_record_for_train_per_node(lines):
    # pattern = r'(.*?)\t(.*?)\(best epoch (.*?), current epoch (.*?)\):(.*?)\.$'
    # pattern = r'(.*?)\tTest at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?)$'
    pattern = r'(.*?)\tGlobal performance per node for train at comm: (.*?) acc: \[(.*?)\]$'
    

    # lines = [line for line in lines if 'best epoch' in line]
    lines = [line for line in lines if 'Global performance per node' in line]
    data = _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_PER_NODE)
    result = [[float(x) for x in row[2].split(', ')] for row in data]
    return result


def _parse_record_for_val(lines):
    # pattern = r'(.*?)\tVal at batch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?)$'
    pattern = r'(.*?)\tVal at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?) Loss: (.*?) Comm: (.*?)$'

    lines = [line for line in lines if 'Val at batch' in line]
    return _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_VAL)

def _parse_record_for_weights(lines):
    # pattern = r'(.*?)\tVal at batch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?)$'
    pattern = r'(.*?)\tAverage norm of models distance from average model is: (.*?)$'

    lines = [line for line in lines if 'Average norm of models' in line]
    return _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_WEIGHT)


def _get_console_record(path):
    # load record file and parse args.
    lines = read_txt(path)

    # parse records.
    parsed_train_lines = _parse_record_for_train(lines)
    parsed_test_lines = _parse_record_for_test(lines)
    parsed_val_lines = _parse_record_for_val(lines)
    parsed_weight_lines = _parse_record_for_weights(lines)
    return parsed_train_lines, parsed_test_lines, parsed_val_lines, parsed_weight_lines


def get_console_records(paths):
    return [_get_console_record(path) for path in paths]

def _save_records_to_csv(source_path, dest_dir=None, test_save=True, val_save=False, weight_save=False, train_save=True):
    if not dest_dir:
        dest_dir = os.path.dirname(source_path)
    records = get_console_records([source_path])
    if train_save:
        train_path = os.path.join(dest_dir, os.path.basename(source_path) + '_train.csv')
        pd.DataFrame(np.array(records[0][0]), columns=VAR_NAMES_TRAIN).to_csv(train_path)
    if test_save:
        test_path = os.path.join(dest_dir, os.path.basename(source_path) + '_test.csv')
        pd.DataFrame(np.array(records[0][1]), columns=VAR_NAMES_TEST).to_csv(test_path)
        if val_save:
            val_path = os.path.join(dest_dir, os.path.basename(source_path) + '_val.csv')
            pd.DataFrame(np.array(records[0][2]), columns=VAR_NAMES_VAL).to_csv(val_path)
    if weight_save:
        weight_path = os.path.join(dest_dir, os.path.basename(source_path) + '_weight.csv')
        pd.DataFrame(np.array(records[0][3]), columns=VAR_NAMES_WEIGHT).to_csv(weight_path)
    print('Records saved!')


def save_records_to_csv(ckpt_dir, num_workers=1, dest_dir=None, test_save='all',
                         val_save=False, server=0, weight=False, train_save=True):
    if not dest_dir:
        dest_dir = ckpt_dir
    for i in range(num_workers):
        source_path = os.path.join(ckpt_dir,str(i),'record{}'.format(i))
        if test_save == 'all':
            _save_records_to_csv(source_path, dest_dir, weight_save=weight, train_save=train_save)
        elif test_save == 'server':
            if i == server:
                _save_records_to_csv(source_path, dest_dir, val_save=val_save, weight_save=weight,train_save=train_save)
            else:
                _save_records_to_csv(source_path, dest_dir, test_save=False, weight_save=weight, train_save=train_save)
        elif test_save == 'none':
            _save_records_to_csv(source_path, dest_dir, test_save=False, weight_save=weight,train_save=train_save)

def save_records_to_csv_new(ckpt_dir, val_save=False, server=0, train_save=True, test_save=False, comm_time=False, per_class=False, per_node=False):
    source_path = os.path.join(ckpt_dir,"0",'record0')
    lines = read_txt(source_path)

    if train_save:
        data1 = _parse_record_for_personal(lines, personal=False, val=False)
        if comm_time:
            cd = parse_record_for_comm_time(ckpt_dir)
            data1 = np.hstack((np.array(data1),cd))
            col = VAR_NAMES_VAL + ['comm_time']
        else:
            data1 = np.array(data1)
            col = VAR_NAMES_VAL
        if per_node:
            pn_data = _parse_record_for_train_per_node(lines)
            data1 = np.hstack((data1,np.array(pn_data)))
            col = col + ['acc_{}'.format(x) for x in range(len(pn_data[0]))]

        path1 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_train.csv')
        pd.DataFrame(data1, columns=col).to_csv(path1)

    if val_save:
        data2 = _parse_record_for_personal(lines, personal=False, val=True)
        path2 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_val.csv')
        pd.DataFrame(np.array(data2), columns=VAR_NAMES_VAL).to_csv(path2)

    if test_save:
        data5 = _parse_record_for_test(lines)
        if per_class:
            pc_data = _parse_record_for_test_per_class(lines)
            data5 = np.hstack((np.array(data5), np.array(pc_data)))
            col = VAR_NAMES_TEST + ['acc_{}'.format(x) for x in range(len(pc_data[0]))]
        else:
            data5 = np.array(data5)
            col = VAR_NAMES_TEST
        path5 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_test.csv')
        pd.DataFrame(data5, columns=col).to_csv(path5)

def save_records_to_csv_personal(ckpt_dir, test=True, val=True):
    source_path = os.path.join(ckpt_dir,"0",'record0')
    lines = read_txt(source_path)
    data1 = _parse_record_for_personal(lines, personal=False, val=False)
    path1 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_np_nv.csv')
    pd.DataFrame(np.array(data1), columns=VAR_NAMES_VAL).to_csv(path1)

    if val:
        data2 = _parse_record_for_personal(lines, personal=False, val=True)
        path2 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_np_v.csv')
        pd.DataFrame(np.array(data2), columns=VAR_NAMES_VAL).to_csv(path2)

    data3 = _parse_record_for_personal(lines, personal=True, val=False)
    path3 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_p_nv.csv')
    pd.DataFrame(np.array(data3), columns=VAR_NAMES_VAL).to_csv(path3)

    if val:
        data4 = _parse_record_for_personal(lines, personal=True, val=True)
        path4 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_p_v.csv')
        pd.DataFrame(np.array(data4), columns=VAR_NAMES_VAL).to_csv(path4)

    if test:
        data5 = _parse_record_for_test(lines)
        path5 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_test.csv')
        pd.DataFrame(np.array(data5), columns=VAR_NAMES_TEST).to_csv(path5)

def save_records_to_csv_per_client_stat(ckpt_dir, test=True, val=True, personal=True):
    source_path = os.path.join(ckpt_dir,"0",'record0')
    lines = read_txt(source_path)
    if not val and not personal:
        data1 = _parse_record_for_per_client_stat(lines, personal=False, val=False)
        path1 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_np_nv_stat.csv')
        pd.DataFrame(np.array(data1), columns=VAR_NAMES_STAT).to_csv(path1)

    if not personal and val:
        data2 = _parse_record_for_per_client_stat(lines, personal=False, val=True)
        path2 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_np_v_stat.csv')
        pd.DataFrame(np.array(data2), columns=VAR_NAMES_STAT).to_csv(path2)

    if personal and not val:
        data3 = _parse_record_for_per_client_stat(lines, personal=True, val=False)
        path3 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_p_nv_stat.csv')
        pd.DataFrame(np.array(data3), columns=VAR_NAMES_STAT).to_csv(path3)

    if personal and val:
        data4 = _parse_record_for_per_client_stat(lines, personal=True, val=True)
        path4 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_p_v_stat.csv')
        pd.DataFrame(np.array(data4), columns=VAR_NAMES_STAT).to_csv(path4)

    if test:
        data5 = _parse_record_for_test_stat(lines)
        path5 = os.path.join(ckpt_dir, os.path.basename(source_path) + '_test_stat.csv')
        pd.DataFrame(np.array(data5), columns=VAR_NAMES_STAT).to_csv(path5)


def _parse_record_for_personal(lines, personal=False, val=False):
    # pattern = r'(.*?)\tVal at batch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?)$'
    if personal:
        if val:
            pattern = r'(.*?)\tPersonal performance for validation at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?) Loss: (.*?) Comm: (.*?)$'
            p = "Personal performance for validation"
        else:
            pattern = r'(.*?)\tPersonal performance for train at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?) Loss: (.*?) Comm: (.*?)$'
            p = "Personal performance for train"
    else:
        if val:
            pattern = r'(.*?)\tGlobal performance for validation at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?) Loss: (.*?) Comm: (.*?)$'
            p = "Global performance for validation"
        else:
            pattern = r'(.*?)\tGlobal performance for train at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?) Loss: (.*?) Comm: (.*?)$'
            p = "Global performance for train"

    lines = [line for line in lines if p in line]
    return _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_VAL)

def _parse_record_for_per_client_stat(lines, personal=False, val=False):
    # pattern = r'(.*?)\tVal at batch: (.*?)\. Process: (.*?)\. Prec@1: (.*?) Prec@5: (.*?)$'
    if personal:
        if val:
            pattern = r'(.*?)\tPersonal per client stat for validation at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Worst: (.*?) Best: (.*?) Var: (.*?) Comm: (.*?)$'
            p = "Personal per client stat for validation"
        else:
            pattern = r'(.*?)\tPersonal per client stat for train at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Worst: (.*?) Best: (.*?) Var: (.*?) Comm: (.*?)$'
            p = "Personal per client stat for train"
    else:
        if val:
            pattern = r'(.*?)\tGlobal per client stat for validation at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Worst: (.*?) Best: (.*?) Var: (.*?) Comm: (.*?)$'
            p = "Global per client stat for validation"
        else:
            pattern = r'(.*?)\tGlobal per client stat for train at batch: (.*?)\. Epoch: (.*?)\. Process: (.*?)\. Worst: (.*?) Best: (.*?) Var: (.*?) Comm: (.*?)$'
            p = "Global per client stat for train"

    lines = [line for line in lines if p in line]
    return _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_STAT)

def parse_record_for_alpha(ckpt_dir):
    source_path = os.path.join(ckpt_dir,"0",'record0')
    lines = read_txt(source_path)
    pattern = r'(.*?)\tNew alpha is:(.*?)$'
    lines = [line for line in lines if 'New alpha' in line]
    data = _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_ALPHA)
    path = os.path.join(ckpt_dir, os.path.basename(source_path) + '_alpha.csv')
    pd.DataFrame(np.array(data), columns=VAR_NAMES_ALPHA).to_csv(path)

def parse_record_for_comm_time(ckpt_dir):
    source_path = os.path.join(ckpt_dir,"0",'record0')
    lines = read_txt(source_path)
    pattern = r'(.*?)\tThis round communication time is: (.*?)$'
    lines = [line for line in lines if 'communication time' in line]
    data = _parse_record(lines, _parse_record_for_test_fn, pattern, VAR_NAMES_COMM)
    return np.expand_dims(np.array(data)[:,1],1)



def get_checkpoint_args(path):
    checkpoint = torch.load(path, map_location='cpu')
    arguments = vars(checkpoint['arguments'])

    arguments.update({
        'n_nodes': arguments['graph'].n_nodes,
        'world': arguments['graph'].world,
        'rank': arguments['graph'].rank,
        'ranks': arguments['graph'].ranks,
        'ranks_with_blocks': arguments['graph'].ranks_with_blocks,
        'blocks_with_ranks': arguments['graph'].blocks_with_ranks,
        'device': arguments['graph'].device,
        'on_cuda': arguments['graph'].on_cuda,
    })
    arguments['graph'] = None
    return arguments
