import os.path

import torch
import numpy as np
import random
import datetime
import pandas as pd


def print_running_info(args):
    print(f'current setting: task_type:{args.task_type},lr:{args.lr},batch_size:{args.train_batch_size}')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def CEloss(logits, labels):
    bsz = logits.shape[0]
    return -torch.log(logits[range(bsz), labels]).mean()


def MSEloss(pred, golden):
    return torch.mean((pred - golden) ** 2)



def get_trading_day_list(history_df):
    day_set = list(set(history_df.date.to_list()))
    day_list = []
    for day in day_set:
        day_time = datetime.datetime.strptime(day, '%Y-%m-%d')
        day_list.append(day_time)
    day_list = sorted(day_list)
    return day_list


def find_nearest_next_trading_day(day_list, cur_date):
    if day_list == []:
        return None
    if cur_date > day_list[-1]:
        return None
    while (cur_date <= day_list[-1]):
        if cur_date in day_list:
            return cur_date
        cur_date = cur_date + datetime.timedelta(days=1)
    return None


def find_delta_trading_day(day_list, cur_date, delta):
    '''
    :param day_set:
    :param cur_date: datetime
    :return:
    '''
    idx = day_list.index(cur_date)
    res = idx + delta
    if res >= len(day_list) or res < 0:
        return None
    return day_list[res]




if __name__ == "__main__":
    pass

