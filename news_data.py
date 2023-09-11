import os
import csv
import json
import sys
import datetime
import numpy as np
import pandas as pd
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from openprompt.data_utils import InputExample, InputFeatures
from utils import get_trading_day_list, find_delta_trading_day, find_nearest_next_trading_day
from torch.utils.data._utils.collate import default_collate


class StockDataWithNews():
    def __init__(self, task_type, cache_dir, code_list, topix_dir, news_path, start_time, end_time):
        # super().__init__()
        self.past_day_num = 20  # data used for prediction
        self.data_time_range = '30min'

        self.raw_data = []
        self.code_list = code_list
        self.topix_dir = topix_dir
        self.news_path = news_path

        self.task_type = task_type
        assert self.task_type in ['volume']
        assert self.data_time_range in ['30min']

        # the range for the news take time [start_time,end_time)

        save_path = os.path.join(cache_dir,
                                 'HardPromptWithData' + self.task_type + '_' + self.data_time_range + '_'
                                 + start_time + '_' + end_time + '_data.pt')

        self.start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d')
        self.end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d')

        if os.path.exists(save_path):
            print('loading data...')
            self.raw_data = torch.load(save_path)
            print('loading finished!')
        else:
            print('generating data...')
            self.load_all_data()
            torch.save(self.raw_data, save_path)
            print('data generated and saved.')

        # self.data=self.raw_data

        # print('generating data...')
        # self.load_all_data()
        # torch.save(self.data, save_path)
        # print('data generated and saved.')

    def load_all_data(self):
        self.topix_list = []
        for filename in os.listdir(self.topix_dir):
            if filename[0] == '.':
                pass
            self.topix_list.append(filename.split('.')[0])
        self.load_news_headline()
        self.add_data()

    def load_news_headline(self):
        with open(self.news_path, 'r') as f:
            self.news_data = json.load(f)

    def add_data(self):
        idx = 0
        for code in tqdm.tqdm(self.code_list):
            if code in self.news_data and code in self.topix_list:
                # TODO: extract more history data here
                history_df = pd.read_csv(os.path.join(self.topix_dir, code + '.csv'))  # history dataframe
                history_df['date_time'] = history_df.date + ' ' + history_df.time

                history_df = history_df.set_index('date_time')  # index:date_time e.g: 2013-01-04 09:00
                trading_day_list = get_trading_day_list(history_df)

                # take_time_str e.g. 2018-03-03; the date is the time the news is taken
                for take_time_str in self.news_data[code]:
                    take_time = datetime.datetime.strptime(take_time_str, '%Y-%m-%d')
                    if take_time < self.start_time or take_time >= self.end_time or take_time not in trading_day_list:
                        continue
                    # pred_time = take_time + datetime.timedelta(days=1)
                    pred_time = find_delta_trading_day(trading_day_list, take_time, delta=1)
                    if pred_time == None:
                        continue

                    res = self.get_volume_data(history_df, trading_day_list, pred_time)
                    if res == None:  # the data should be filtered
                        continue
                    history, gold_volume, label = res

                    # TODO:process text data here
                    # simply join all headlines
                    headlines = '. '.join(self.news_data[code][take_time_str])
                    headlines = headlines.lower()

                    cur_instance = InputExample(text_a=headlines, label=label, guid=idx)
                    idx += 1

                    self.raw_data.append((cur_instance, history, gold_volume))

    def get_30min_volume_data(self, history_df, trading_day_list, pred_date):
        pred_time = pred_date + datetime.timedelta(hours=9)
        pred_volume = 0
        for min_slot in range(6):
            cur_time = pred_time + datetime.timedelta(minutes=min_slot * 5)
            pred_time_str = cur_time.strftime('%Y-%m-%d %H:%M')
            if pred_time_str not in history_df.index:
                return None
            pred_volume += history_df.loc[pred_time_str, 'v']
        gold_volume = float(pred_volume)

        history = torch.zeros((self.past_day_num, 10, 5))  # [20days; 10 slots; 5 features: [c,h,l,o,v]]
        history[:, :, 2] = 1e8  # init low price

        for day in range(0, self.past_day_num):
            # cur_date = pred_date + datetime.timedelta(days=-20 + day)
            cur_date = find_delta_trading_day(trading_day_list, pred_date, delta=-20 + day)
            if cur_date == None:
                return None

            cur_morning_time = cur_date + datetime.timedelta(hours=9)
            cur_afternoon_time = cur_date + datetime.timedelta(hours=12, minutes=30)
            for half_hour_slot in range(10):
                # 0:[9:00,9:30),1:[9:30,10:00),2:[10:00,10:30),3:[10:30,11:00),4:[11:00,11:30]
                if half_hour_slot < 5:
                    cur_slot_time = cur_morning_time + datetime.timedelta(minutes=half_hour_slot * 30)
                # 5:[12:30,13:00),6:[13:00,13:30),7:[13:30,14:00),8:[14:00,14:30),9:[14:30,15:00]
                else:
                    cur_slot_time = cur_afternoon_time + datetime.timedelta(minutes=(half_hour_slot - 5) * 30)

                # if it is [11:00,11:30] or [14:30,15:00], there will be an additional minunte slot
                min_slot_num = 6 + ((half_hour_slot % 5) == 4)
                for min_slot in range(min_slot_num):
                    cur_time = cur_slot_time + datetime.timedelta(minutes=min_slot * 5)
                    cur_time_str = cur_time.strftime('%Y-%m-%d %H:%M')
                    if cur_time_str not in history_df.index:
                        return None
                    # closing
                    if min_slot == min_slot_num - 1:
                        history[day, half_hour_slot, 0] = history_df.loc[cur_time_str, 'c']

                    # high
                    history[day, half_hour_slot, 1] = max(history[day, half_hour_slot, 1],
                                                          history_df.loc[cur_time_str, 'h'])

                    # low
                    history[day, half_hour_slot, 2] = min(history[day, half_hour_slot, 1],
                                                          history_df.loc[cur_time_str, 'l'])
                    # opening
                    if min_slot == 0:
                        history[day, half_hour_slot, 3] = history_df.loc[cur_time_str, 'o']

                    # volume
                    history[day, half_hour_slot, 4] += history_df.loc[cur_time_str, 'v']

        # volumes = history[:, :, 4].view(-1)
        volumes = history[:, 0, 4]
        v_mean = torch.mean(volumes)
        v_std = torch.std(volumes, unbiased=False)
        y = (gold_volume - v_mean) / v_std

        label = 0
        if y > 0.5:
            label = 0
        elif y < -0.5:
            label = 1
        else:
            return None

        return history, gold_volume, label

    def get_volume_data(self, history_data, trading_day_list, pred_date):
        if self.data_time_range == '30min':
            return self.get_30min_volume_data(history_data, trading_day_list, pred_date)
        else:
            raise NotImplementedError

    def process_data(self, tokenizer):
        self.processed_data = []
        for item in self.raw_data:
            example, history, golden_volume = item
            sen, label = example.text_a, example.label
            inputs = tokenizer(text=sen, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            new_inputs = {}
            for key in inputs.keys():
                new_inputs[key] = inputs[key].squeeze(dim=0)
            self.processed_data.append((new_inputs, history, golden_volume, label))

    def get_dataloader(self, batch_size, shuffle=False, drop_last=False):
        assert len(self.processed_data) > 1
        return DataLoader(self.processed_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                          collate_fn=self.collate_fct)


    def show_info(self, prompt):
        print(f'{prompt} total_data length:{len(self.raw_data)}')

    @staticmethod
    def collate_fct(batch):
        elem = batch[0][0]
        return_dict = {}
        for key in elem:
            if key == "encoded_tgt_text":
                return_dict[key] = [d[0][key] for d in batch]
            else:
                return_dict[key] = default_collate([d[0][key] for d in batch])

        history = torch.stack([d[1].view(200, 5) for d in batch], dim=0)
        golden_volume = torch.stack([torch.tensor(d[2]).view(1) for d in batch], dim=0)
        label = torch.cat([torch.tensor(d[3]).view(1) for d in batch], dim=0).long()
        history = torch.log(history)
        golden_volume = torch.log(golden_volume)

        return return_dict, history, golden_volume, label


class HistoryData(Dataset):
    def __init__(self, task_type, cache_dir, code_list, topix_dir, start_time, end_time):
        self.past_day_num = 20  # data used for prediction
        self.data_time_range = '30min'

        # List of tensors with shape:  [batch_len,10,5] note that batch_len for each tensor can be different
        self.data = []
        self.task_type = task_type
        self.code_list = code_list
        self.topix_dir = topix_dir

        assert self.task_type in ['volume']
        assert self.data_time_range in ['30min']

        self.start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d')
        self.end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d')

        save_path = os.path.join(cache_dir,
                                 'History_data' + self.task_type + '_' + self.data_time_range + '_'
                                 + start_time + '_' + end_time + '_data.pt')

        if os.path.exists(save_path):
            print('loading data...')
            self.data = torch.load(save_path)
            print('loading finished!')
        else:
            print('generating data...')
            self.generate_history_data()
            torch.save(self.data, save_path)
            print('data generated and saved.')

        self.construct_idx2data()

    def generate_history_data(self):
        self.topix_list = []
        for filename in os.listdir(self.topix_dir):
            if filename[0] == '.':
                pass
            self.topix_list.append(filename.split('.')[0])

        for code in tqdm.tqdm(self.code_list):
            if code not in self.topix_list:
                continue

            history_df = pd.read_csv(os.path.join(self.topix_dir, code + '.csv'))  # history dataframe
            history_df['date_time'] = history_df.date + ' ' + history_df.time

            history_df = history_df.set_index('date_time')  # index:date_time e .g: 2013-01-04 09:00
            trading_day_list = get_trading_day_list(history_df)

            self.data += self.get_30min_volume_data(history_df, trading_day_list)

    def construct_idx2data(self):
        self.total_length = 0
        self.idx2data = []
        for batch_num, batch in enumerate(self.data):
            cur_len = batch.shape[0]
            for idx in range(0, cur_len - 20):
                self.idx2data.append((batch_num, idx))
            self.total_length += cur_len - 20

    def show_info(self, prompt):
        print(f'{prompt} total_data length:{len(self)}')

    def get_30min_volume_data(self, history_df, trading_day_list):
        history_res = []

        # init
        cur_history_batch = []
        cur_date = find_nearest_next_trading_day(trading_day_list, self.start_time)

        break_flag = 0

        # because the data of the last day is only used for prediction, so it is <= here.
        while (cur_date <= self.end_time):
            cur_history = torch.zeros((10, 5))  # [10 slots; 5 features: [c,h,l,o,v]]]
            cur_history[:, 2] = 1e8  # init low prices

            cur_date = find_delta_trading_day(trading_day_list, cur_date, delta=1)
            if cur_date == None:  # no more data
                break_flag = 1
                cur_date = self.end_time + datetime.timedelta(days=1)  # to break from while
            else:
                cur_morning_time = cur_date + datetime.timedelta(hours=9)
                cur_afternoon_time = cur_date + datetime.timedelta(hours=12, minutes=30)
                for half_hour_slot in range(10):
                    # 0:[9:00,9:30),1:[9:30,10:00),2:[10:00,10:30),3:[10:30,11:00),4:[11:00,11:30]
                    if half_hour_slot < 5:
                        cur_slot_time = cur_morning_time + datetime.timedelta(minutes=half_hour_slot * 30)
                    # 5:[12:30,13:00),6:[13:00,13:30),7:[13:30,14:00),8:[14:00,14:30),9:[14:30,15:00]
                    else:
                        cur_slot_time = cur_afternoon_time + datetime.timedelta(minutes=(half_hour_slot - 5) * 30)

                    # if it is [11:00,11:30] or [14:30,15:00], there will be an additional minunte slot
                    min_slot_num = 6 + ((half_hour_slot % 5) == 4)
                    for min_slot in range(min_slot_num):
                        cur_time = cur_slot_time + datetime.timedelta(minutes=min_slot * 5)
                        cur_time_str = cur_time.strftime('%Y-%m-%d %H:%M')
                        if cur_time_str not in history_df.index:
                            break_flag = 1
                            break
                        # closing
                        if min_slot == min_slot_num - 1:
                            cur_history[half_hour_slot, 0] = history_df.loc[cur_time_str, 'c']

                        # high
                        cur_history[half_hour_slot, 1] = max(cur_history[half_hour_slot, 1],
                                                             history_df.loc[cur_time_str, 'h'])

                        # low
                        cur_history[half_hour_slot, 2] = min(cur_history[half_hour_slot, 1],
                                                             history_df.loc[cur_time_str, 'l'])
                        # opening
                        if min_slot == 0:
                            cur_history[half_hour_slot, 3] = history_df.loc[cur_time_str, 'o']

                        # volume
                        cur_history[half_hour_slot, 4] += history_df.loc[cur_time_str, 'v']

                    if break_flag:
                        break

            if break_flag:  # the end of a batch because missing data detected
                break_flag = 0
                if len(cur_history_batch) >= 21:  # current batch is long enough
                    history_res.append(torch.stack(cur_history_batch, dim=0))
                cur_history_batch = []
            else:
                cur_history_batch.append(cur_history)

        return history_res

    def __len__(self):
        return self.total_length

    def __getitem__(self, item):
        batch_num, idx = self.idx2data[item]
        history = self.data[batch_num][idx:idx + 20]  # [20,10,5]
        gold_volume = self.data[batch_num][idx + 20, 0, 4]  # [1]

        return (history, gold_volume)


def extract_news(dir, code_list):
    csv.field_size_limit(sys.maxsize)
    years = ['2013', '2014', '2015', '2016', '2017', '2018']
    company_set = set(code_list)
    company_news = {company: dict() for company in company_set}
    news_number = {company: 0 for company in company_set}
    for year in years:
        print(f'processing year {year}...')
        year_dir = os.path.join(dir, year)
        if not os.path.isdir(year_dir) or int(year) < 2013:
            continue
        months = os.listdir(year_dir)
        for month in tqdm.tqdm(months):
            month_dir = os.path.join(dir, year, month)
            if not os.path.isdir(month_dir):
                continue
            dates = os.listdir(month_dir)
            for date in dates:
                csv_name = os.path.join(month_dir, date, year + '-' + month + '-' + date + '.csv')
                with open(csv_name) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
                    for row in csv_reader:
                        id, _, story_time, take_time, headline, body, product, topic, rics, _, language, _, _, _, _, _, _ = row
                        if language != 'en':
                            continue
                        # if 'price' not in headline:
                        #     continue
                        take_time = datetime.datetime.strptime(take_time, '%m/%d/%Y %H:%M:%S')
                        # only use news happened between UTC+9 15:00 to 09:00+1day (news happened when the market is off)
                        # corresponding to UTC+0 6:00 to 24:00
                        if take_time.hour >= 6:
                            take_time_date_str = take_time.date().strftime('%Y-%m-%d')
                        else:
                            # the market is open;
                            continue
                        for ric in rics.split(' '):
                            tem = ric.split('.')
                            if len(tem) >= 2:
                                code = tem[0]
                                market = tem[1]
                            else:
                                code = 'None'
                                market = tem[0]
                            if code in company_set and market == 'T':
                                if take_time_date_str not in company_news[code]:
                                    company_news[code][take_time_date_str] = []
                                # company_news[code][take_time_date_str].append(
                                #     (id, headline, body, topic, rics, take_time.time().strftime('%H')))

                                company_news[code][take_time_date_str].append(headline)
                                news_number[code] += 1
                            '''
                            elif market == 'N225' or market.startswith('TOPX'):
                                for company in company_news:
                                    if take_time_date_str not in company_news[company]:
                                        company_news[company][take_time_date_str] = []
                                    company_news[company][take_time_date_str].append(
                                        (id, headline, body, topic, rics, take_time.time().strftime('%H')))
                            '''

    print('average price news number', sum(news_number.values()) / float(len(news_number)))
    with open(os.path.join(dir, 'company_news_full.json'), 'w') as f:
        json.dump(company_news, f)

def extract_stock_codes(stock_data_path='./data/stock_data.csv'):
    stock_data = pd.read_csv(stock_data_path)
    h, w = stock_data.shape
    code_list = []
    for idx in range(h):
        if stock_data.loc[idx, 'group'] in ['CORE30', 'LARGE70', 'MID400']:
            code_list.append(str(stock_data.loc[idx, 'code']))
    return code_list







if __name__ == '__main__':
    pass