# -*- coding: utf-8 -*-
import re
import os

from fedtorch.utils.op_paths import list_files
from fedtorch.tools.load_console_records import get_console_records, get_checkpoint_args

"""load records."""


def parse_record_from_current_folder(folder_path):
    print('process the folder: {}'.format(folder_path))
    checkpoint_path = os.path.join(folder_path, 'checkpoint.pth.tar')

    sub_folder_paths = [
        sub_folder_path for sub_folder_path in list_files(folder_path)
        if '.tar' not in sub_folder_path
    ]

    # collect info for one rank.
    sub_folder_paths = sorted(sub_folder_paths)
    sub_folder_path = sub_folder_paths[1] \
        if len(sub_folder_paths) > 1 else sub_folder_paths[0]
    rank = sub_folder_path.split('/')[-1]
    console_paths = [os.path.join(sub_folder_path, 'record{}'.format(rank))]
    print('    with the specific rank: {}'.format(rank))

    return (
        folder_path, {
            'console_record': get_console_records(console_paths),
            'arguments': get_checkpoint_args(checkpoint_path)
        }
    )


def parse_records(root_path):
    folder_paths = [
        folder_path
        for folder_path in list_files(root_path)
        if '.pickle' not in folder_path
    ]

    info = []
    for folder_path in folder_paths:
        try:
            element_of_info = parse_record_from_current_folder(folder_path)
            info.append(element_of_info)
        except Exception as e:
            print('error: {}'.format(e))
    return info


"""parse the argument from the console records."""


def parse_args_from_console_records(records):
    regex = re.compile(r'.\t.*\t')
    cur_args = [
        tuple(l.split('\t')[1:])
        for l in records[: 1000] if re.search(regex, l)
    ]
    return dict(cur_args)


"""load data from pickled file."""


def load_records(records):
    record_path, _detailed_record = records
    console_record = _detailed_record['console_record']
    arguments = _detailed_record['arguments']
    return arguments, console_record


def extract_interested_args(arguments, interested_args):
    return dict((arg, arguments[arg]) for arg in interested_args)


# extract records.

def _find_index(_list, _value):
    try:
        return _list.index(_value)
    except Exception as e:
        return -1


def _is_same(items):
    return len(set(items)) == 1


def _is_meet_conditions(items):
    item_set = set(items)

    if -1 in item_set:
        return False
    return len(set(items)) == 1


def is_meet_conditions(args, conditions):
    if conditions is None:
        return True

    # get condition values and have a safety check.
    condition_names = list(conditions.keys())
    condition_values = list(conditions.values())
    assert _is_same([len(values) for values in condition_values]) is True

    # re-build conditions.
    num_condition = len(condition_values)
    num_condition_value = len(condition_values[0])
    condition_values = [
        [condition_values[ind_cond][ind_value]
            for ind_cond in range(num_condition)]
        for ind_value in range(num_condition_value)
    ]

    # check re-built condition.
    g_flag = False
    for cond_values in condition_values:
        l_flag = True
        for ind, cond_value in enumerate(cond_values):
            l_flag = l_flag and (cond_value == args[condition_names[ind]])
        g_flag = g_flag or l_flag

    return g_flag


def extract_list_of_records(list_of_records, conditions, interested_args,
                            record_type, use_log=False):
    # load and filter data.
    records = []
    num_updates_and_local_step = []

    for raw_records in list_of_records:
        arguments, console_records = load_records(raw_records)

        # get arguments.
        args = extract_interested_args(arguments, interested_args)

        if use_log:
            print(('\twe are processing {} dataset on {}. We have {} workers, with {} local mini-batch size. The local update step={}. \n' +
                  '\tWe are using constant lr={}, with weight decay={}.\n').format(
                args['data'], args['arch'], args['n_nodes'], args['batch_size'], args['local_step'],
                args['learning_rate'], args['weight_decay']
            ))

        # check conditions.
        if not is_meet_conditions(args, conditions):
            continue

        # get parsed records
        info = parse_records(console_records)
        records += [(args, info)]
        num_updates_and_local_step += [(len(info['tr_steps']), args['local_step'])]

    print('we have {}/{} records.'.format(len(records), len(list_of_records)))
    return records, num_updates_and_local_step


def reorder_records(records, based_on):
    # records is in the form of <args, info>
    conditions = based_on.split(',')
    list_of_args = [
        (ind, [args[condition] for condition in conditions])
        for ind, (args, info) in enumerate(records)
    ]
    sorted_list_of_args = sorted(list_of_args, key=lambda x: x[1:])
    return [records[ind] for ind, args in sorted_list_of_args]


# for console.


def _get_parsed_console_tr_records(events):
    events_loss = [e['loss'] for e in events]
    events_top1 = [e['top1'] for e in events]
    events_time = [e['time'] for e in events]

    num_events = len(events_top1)
    return events_loss, events_top1, events_time, num_events


def _get_parsed_console_te_records(events):
    events_top1 = [e['top1'] for e in events]
    events_times = [e['time'] for e in events]

    num_events = len(events_top1)
    return events_top1, events_times, num_events


def _get_parsed_console_records(console_record):
    tr_info, te_info = console_record

    parsed_tr_events = _get_parsed_console_tr_records(tr_info)
    parsed_te_events = _get_parsed_console_te_records(te_info)
    return parsed_tr_events, parsed_te_events


def get_parsed_console_records(console_records):
    tr_info, te_info = _get_parsed_console_records(console_records[0])
    tr_events_loss, tr_events_top1, tr_events_time, num_tr_events = tr_info
    te_events_top1, te_events_time, te_event_steps = te_info

    return {
        'tr_loss': tr_events_loss,
        'tr_top1': tr_events_top1,
        'tr_steps': list(range(1, 1 + num_tr_events)),
        'tr_time': tr_events_time,
        'te_top1': te_events_top1,
        'te_steps': te_event_steps,
        'te_times': te_events_time
    }
