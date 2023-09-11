import os
import argparse

parser = argparse.ArgumentParser()

# path
parser.add_argument('--code_list_path', default='./data/code_list.json', type=str, help='Stock Symbol')
parser.add_argument('--topix_dir', default='./topix500', type=str, help='Topix dataset')
parser.add_argument('--news_path', default='./data/company_news_full.json', type=str, help='Financial news')
parser.add_argument('--cache_dir', default='./data_cache', type=str, help='Dataset cache')
parser.add_argument('--pre_train_path', default='./model_cache/pretrain.pkl', type=str, help='pre-trained transformer model')

# time spans
parser.add_argument('--train_st', default='2013-01-01', type=str)
parser.add_argument('--train_ed', default='2018-01-01', type=str)
parser.add_argument('--dev_st', default='2018-01-01', type=str)
parser.add_argument('--dev_ed', default='2018-05-01', type=str)
parser.add_argument('--test_st', default='2018-05-01', type=str)
parser.add_argument('--test_ed', default='2018-10-01', type=str)

# model
parser.add_argument('--task_type', default='volume', type=str)
parser.add_argument('--plm_type', default='soleimanian/financial-roberta-large-sentiment', type=str)

#hyperparameters
parser.add_argument('--max_epoch', default=40, type=int)
parser.add_argument('--train_batch_size', default=32, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--alpha_align_loss', default=0.1, type=float)
parser.add_argument('--alpha_news_loss', default=1, type=float)
parser.add_argument('--alpha_data_loss', default=1, type=float)
parser.add_argument('--alpha_cat_loss', default=1, type=float)
parser.add_argument('--temperature', default=1, type=float)

# training setting
parser.add_argument('--seed', default=101, type=int)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--dp', default=0, type=bool)



args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

import torch
import utils
import json
from news_data import StockDataWithNews
from torch import nn
from multi_modal_models import MultimodalModel

def train(model: MultimodalModel, optimizer, train_dataloader):
    model.train()
    tot_loss = torch.zeros(1).cuda()
    step_num = 0
    for inputs, history, golden_volume, labels in train_dataloader:
        history = history.cuda()
        labels = labels.cuda()
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()

        logits_news, logits_data, logits_cat, loss_align = model(input_ids=input_ids, attention_mask=attention_mask,
                                                                 history_data=history)
        loss_news = utils.CEloss(logits_news, labels)
        loss_data = utils.CEloss(logits_data, labels)
        loss_cat = utils.CEloss(logits_cat, labels)
        loss = args.alpha_news_loss * loss_news + args.alpha_data_loss * loss_data + \
               args.alpha_align_loss * loss_align + args.alpha_cat_loss * loss_cat


        tot_loss += loss
        step_num += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    tot_loss /= step_num
    return tot_loss.item()


def test(model: MultimodalModel, test_dataloader):
    model.eval()

    with torch.no_grad():

        correct_num = torch.zeros(1).cuda()
        tot_num = torch.zeros(1).cuda()
        for inputs, history, golden_volume, labels in test_dataloader:
            history = history.cuda()
            labels = labels.cuda()
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            logits_news, logits_data, logits_cat, loss_align = model(input_ids=input_ids,
                                                                     attention_mask=attention_mask,
                                                                     history_data=history)
            logits = args.alpha_news_loss * logits_news + args.alpha_data_loss * logits_data + args.alpha_cat_loss * logits_cat

            pred = logits.argmax(dim=-1)

            tot_num += pred.shape[0]
            correct_num += torch.sum(pred == labels)

        acc = correct_num / tot_num

    return acc.item()


def main():
    utils.set_seed(args.seed)
    with open(args.code_list_path, 'r') as f:
        code_list = json.load(f)

    utils.print_running_info(args)
    print('start loading dataset...')
    train_set = StockDataWithNews(task_type=args.task_type, cache_dir=args.cache_dir, code_list=code_list,
                                  news_path=args.news_path, topix_dir=args.topix_dir, start_time=args.train_st,
                                  end_time=args.train_ed)
    dev_set = StockDataWithNews(task_type=args.task_type, cache_dir=args.cache_dir, code_list=code_list,
                                news_path=args.news_path, topix_dir=args.topix_dir, start_time=args.dev_st,
                                end_time=args.dev_ed)
    test_set = StockDataWithNews(task_type=args.task_type, cache_dir=args.cache_dir, code_list=code_list,
                                 news_path=args.news_path, topix_dir=args.topix_dir, start_time=args.test_st,
                                 end_time=args.test_ed)
    print('dataset loading finished.')

    train_set.show_info('train set')
    dev_set.show_info('dev set')
    test_set.show_info('test set')


    print('start loading model...')
    prefix_config = {'pre_seq_len': 20, 'num_hidden_layers': 24, 'num_attention_heads': 16,
                     'hidden_size': 1024, 'hidden_dropout_prob': 0.1, 'prefix_projection': 1,
                     'prefix_hidden_size': 1024, 'use_return_dict': 0}
    load_pretrain_args = {'path': args.pre_train_path, 'map_location': {'cuda:3': f'cuda:{args.gpu_id}'}}
    model = MultimodalModel(config=prefix_config, roberta_type=args.plm_type,
                            load_pretrain_args=load_pretrain_args,
                            temperature=args.temperature,
                            freeze_plm=True, freeze_trm=False).cuda()

    optim_params = model.parameters()

    optimizer = torch.optim.AdamW(params=optim_params, lr=args.lr, weight_decay=1e-3)
    print('model loading finished.')

    train_set.process_data(tokenizer=model.tokenizer)
    dev_set.process_data(tokenizer=model.tokenizer)
    test_set.process_data(tokenizer=model.tokenizer)

    train_prompt_loader = train_set.get_dataloader(batch_size=args.train_batch_size, shuffle=True,
                                                                 drop_last=True)
    dev_prompt_loader = dev_set.get_dataloader(batch_size=args.test_batch_size)
    test_prompt_loader = test_set.get_dataloader(batch_size=args.test_batch_size)

    print('dataloaders generated.')

    best_dev_acc = 0
    best_test_acc = 0

    for epoch in range(1, args.max_epoch + 1):
        print('-' * 80)
        print(f'running epoch {epoch}')
        if args.dp:
            model = nn.DataParallel(model)
        tot_loss = train(model, optimizer, train_prompt_loader)
        print(f'train loss:{tot_loss}')

        if args.dp:
            model = model.module


        dev_acc = test(model, dev_prompt_loader)
        print(f'dev acc:{dev_acc}')
        test_acc = test(model, test_prompt_loader)
        print(f'test acc:{test_acc}')

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_test_acc = test_acc
    print('-' * 80)
    print('training finished!')
    print(f'test acc on best dev ckpt:{best_test_acc}')


if __name__ == '__main__':
    main()
