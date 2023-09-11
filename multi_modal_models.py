import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from baseline_models import *

class HistoryDataHeader(nn.Module):
    def __init__(self, load_pretrain_args, d_plm, freeze_trm=False):
        super().__init__()
        self.fin_trm = fin_transformer()
        self.load_pretrained_trm(load_pretrain_args)

        if freeze_trm:
            self.fin_trm.freeze_trm_()
        self.d_model = self.fin_trm.trm.d_model
        self.header = nn.Linear(in_features=self.d_model, out_features=d_plm)

    def load_pretrained_trm(self, load_pretrain_args):
        path = load_pretrain_args['path']
        map_location = load_pretrain_args['map_location']
        pretrain_dict = torch.load(path, map_location)
        self.fin_trm.load_state_dict(pretrain_dict)

    def get_cls_vec(self, history):
        return self.fin_trm(history, get_cls=True)

    def get_trm_vec(self, history):
        return self.fin_trm(history, get_emb=True)

    def forward(self, history):
        '''

        :param history: [N,200,5]
        :return:
        '''
        history_emb = self.fin_trm(history, get_emb=True)  # [N,d_trm]
        history_vec = self.header(history_emb)  # [N,d_plm]
        history_vec = history_vec.unsqueeze(dim=1)  # [N,1,d_plm]

        return history_vec


class fin_transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.trm = Transformer()

    def freeze_trm_(self):
        for params in self.trm.parameters():
            params.requires_grad = False

    def forward(self, history, get_emb=False, get_cls=False,get_final_layer=False):
        '''
        :param history: [N,200,5]
        :return:
        '''
        return self.trm(history, early_exit=get_emb, get_cls=get_cls,get_final_layer=get_final_layer)


# from https://github.com/THUDM/P-tuning-v2
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config['prefix_projection']
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config['pre_seq_len'], config['hidden_size'])
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config['hidden_size'], config['prefix_hidden_size']),
                torch.nn.Tanh(),
                torch.nn.Linear(config['prefix_hidden_size'], config['num_hidden_layers'] * 2 * config['hidden_size'])
            )
        else:
            self.embedding = torch.nn.Embedding(config['pre_seq_len'],
                                                config['num_hidden_layers'] * 2 * config['hidden_size'])

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class ClsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(200, config['hidden_size']),
            torch.nn.Tanh(),
            torch.nn.Linear(config['hidden_size'], config['num_hidden_layers'] * 2 * config['hidden_size'])
        )

    def forward(self, cls_vecs):
        '''
        :param cls_vecs: [bsz,6,200]
        :return:
        '''
        past_key_values = self.trans(cls_vecs)
        return past_key_values



class MultimodalModel(nn.Module):
    def __init__(self, config, roberta_type, load_pretrain_args,  temperature, freeze_plm=False,
                 freeze_trm=False):
        super().__init__()
        self.config = config

        # history header's trm is frozen
        self.history_header = HistoryDataHeader(load_pretrain_args, d_plm=1024, freeze_trm=freeze_trm)

        if roberta_type == 'soleimanian/financial-roberta-large-sentiment':
            self.tokenizer = RobertaTokenizer.from_pretrained('./pretrained_models/financial-roberta-large-sentiment')
            self.roberta = RobertaModel.from_pretrained('./pretrained_models/financial-roberta-large-sentiment')
        else:
            raise NotImplementedError


        if freeze_plm:
            for params in self.roberta.parameters():
                params.requires_grad = False

        self.dropout = torch.nn.Dropout(self.config['hidden_dropout_prob'])

        self.pre_seq_len = config['pre_seq_len']
        self.n_layer = config['num_hidden_layers']
        self.n_head = config['num_attention_heads']
        self.n_embd = config['hidden_size'] // config['num_attention_heads']

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)
        self.cls_encoder = ClsEncoder(config)

        self.hidden_dim = 200
        self.trm_proj = nn.Linear(self.history_header.d_model, self.hidden_dim)
        self.plm_proj = nn.Linear(config['hidden_size'], self.hidden_dim)

        self.plm_classifier = nn.Linear(config['hidden_size'], 2)
        self.trm_classifier = nn.Linear(self.history_header.d_model, 2)
        self.cat_classifier = nn.Linear(2 * self.hidden_dim, 2)

        self.temp = temperature  # temperature

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        # [bsz,pre_seq_len,2*n_layer,self.n_head,self.n_embed]
        past_key_values = self.dropout(past_key_values)

        # [2*n_layer,bsz,self.n_head,pre_seq_len,self.n_embed]
        # [n_layer*[2,bsz,self.n_head,pre_seq_len,self.n_embed]]]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            history_data=None
    ):

        return_dict = return_dict if return_dict is not None else self.config['use_return_dict']
        batch_size = input_ids.shape[0]
        prompt_past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        past_key_values = prompt_past_key_values

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        data_output = self.history_header.get_trm_vec(history_data)  # [bsz,200]
        news_output = outputs[0][:, 0, :]  # [bsz,1024]
        data_h = torch.tanh(self.trm_proj(data_output))  # [bsz,h]
        news_h = torch.tanh(self.plm_proj(news_output))  # [bsz,h]

        sim = (news_h @ data_h.T) / self.temp  # [bsz,bsz] sim_ij=news_h_i*data_h_j
        bsz = sim.shape[0]

        loss_align = -torch.log(
            torch.softmax(sim, dim=1)[torch.arange(bsz).cuda(), torch.arange(bsz).cuda()]).mean() - torch.log(
            torch.softmax(sim, dim=0)[torch.arange(bsz).cuda(), torch.arange(bsz).cuda()]).mean()

        logits_news = torch.softmax(self.plm_classifier(news_output), dim=-1)  # [bsz,2]
        logits_data = torch.softmax(self.trm_classifier(data_output), dim=-1)  # [bsz,2]
        logits_cat = torch.softmax(self.cat_classifier(torch.cat([news_h, data_h], dim=-1)), dim=-1)  # [bsz,2]
        return logits_news, logits_data, logits_cat, loss_align