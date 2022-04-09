# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from DynamicRNN import DynamicRNN
from myutils import vectors2padsequence
from attention import Attention, AttentionTri, Attention_Modified

class dualModel(nn.Module):
    def __init__(self, opt, n_vocab, embed_list):
        super(dualModel, self).__init__()
        self.n_layers = opt.n_layers
        self.dim_input = opt.dim_input
        self.dim_hidden = opt.dim_hidden
        self.bidirecitional = True if opt.bidirectional else False  # int->bool
        self.rnn_type = opt.rnn_type
        self.device = opt.device
        self.multi_dim = opt.multi_dim
        self.max_length_sen = opt.max_length_sen
        n_hidden = 128
        n_hidden_literal = 64
        self.dim_profile = 64
        dim_sen = opt.dim_hidden * (2 if self.bidirecitional else 1)
        self.t_sne = opt.t_sne # always together with predict mode

        # self.add_module('embed', nn.Embedding(n_vocab, self.dim_input))
        # self.add_module('embed_dropout', nn.Dropout(opt.embed_dropout_rate))
        self.add_module('literal', BiRNN_dualModel(opt, n_vocab, embed_list))
        self.add_module('deep', BiRNN_dualModel(opt, n_vocab, embed_list))
        self.add_module('sen', BiRNN_dualModel(opt, n_vocab, embed_list))
        self.add_module('reduce_literal', nn.Linear(dim_sen*2, dim_sen))
        self.add_module('reduce_deep', nn.Linear(dim_sen*2, dim_sen))
        self.add_module('relu', nn.ReLU())

        self.add_module('dense_literal', nn.Linear(dim_sen, opt.n_class))
        self.add_module('dense_deep', nn.Linear(dim_sen, opt.n_class))
        # self.add_module('project', nn.Linear(dim_sen, n_hidden)) # bias=False
        self.add_module('dense', nn.Linear(dim_sen*2, opt.n_class))

        # self.init_weights(embed_list)
        # self.ignored_params = self.sen.embed.parameters() + self.literal.embed.parameters() + self.deep.embed.parameters() # generator cannot be added.

        params_senti = list(map(id, self.literal.embed.parameters()))
        params_nonsenti = list(map(id, self.deep.embed.parameters()))
        params_sen = list(map(id, self.sen.embed.parameters()))
        self.base_params = list(
            filter(lambda p: id(p) not in (params_sen + params_senti + params_nonsenti),
                   self.parameters()))
        print('****parameter all set****')

    # def init_weights(self, embed_list):
    #     self.embed.weight.data.copy_(torch.from_numpy(embed_list))


    def forward(self, dict_inst):
        # senti = self.embed_dropout(self.embed(dict_inst['senti_sens']))
        # nonsenti = self.embed_dropout(self.embed(dict_inst['nonsenti_sens']))
        # sen = self.embed_dropout(self.embed(dict_inst['sens']))

        # non project for getting a representation
        literal_rep = self.literal(dict_inst['senti_sens'], dict_inst['senti_len_sen'])
        deep_rep = self.deep(dict_inst['nonsenti_sens'], dict_inst['nonsenti_len_sen'])
        sen_rep = self.sen(dict_inst['sens'], dict_inst['len_sen'])

        # project
        literal_rep_proj = self.reduce_literal(torch.cat([literal_rep, sen_rep], dim=-1))
        deep_rep_proj = self.reduce_deep(torch.cat([deep_rep, sen_rep], dim=-1))
        literal_rep_proj = self.relu(literal_rep_proj)
        deep_rep_proj = self.relu(deep_rep_proj)

        prob_senti = torch.softmax(self.dense_literal(literal_rep), dim=-1)
        prob_nonsenti = torch.softmax(self.dense_deep(deep_rep), dim=-1)

        # analyzer module
        dense_input = torch.cat([literal_rep_proj, deep_rep_proj], dim=-1)

        prob = torch.softmax(self.dense(dense_input), dim=-1)
        if self.t_sne:
            return prob, prob_senti, prob_nonsenti, literal_rep_proj, deep_rep_proj, dense_input # literal=deep!=sen different dimension
            # return prob, prob_senti, prob_nonsenti, literal_rep, deep_rep, sen_rep # same dimension
        else:
            return prob, prob_senti, prob_nonsenti

class BiRNN_dualModel(nn.Module):
    '''
    Decoding the sentences in feedbacks
    Inout: sentences
    Output: sentence vectors, feedback vector
    '''

    def __init__(self, opt, n_vocab, embed_list):
        super(BiRNN_dualModel, self).__init__()
        self.n_layers = opt.n_layers
        self.dim_input = opt.dim_input
        self.dim_hidden = opt.dim_hidden
        self.bidirecitional = True if opt.bidirectional else False  # int->bool
        self.rnn_type = opt.rnn_type
        self.device = opt.device

        self.add_module('embed', nn.Embedding(n_vocab, self.dim_input))
        self.add_module('embed_dropout', nn.Dropout(opt.embed_dropout_rate))
        self.add_module('rnn_sen', DynamicRNN(opt.dim_input, opt.dim_hidden, opt.n_layers,
                                           dropout=(opt.cell_dropout_rate if opt.n_layers > 1 else 0),
                                           bidirectional=self.bidirecitional, rnn_type=opt.rnn_type, device=opt.device))
        self.init_weights(embed_list)
        # self.ignored_params_id = list(map(id, self.embed.parameters()))
        # self.base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))

    def init_weights(self, embed_list):
        self.embed.weight.data.copy_(torch.from_numpy(embed_list))

    def forward(self, embed_list, length_list):
        embedded = self.embed_dropout(self.embed(embed_list))

        output_pad, hidden_encoder = self.rnn_sen(embedded, lengths=length_list, flag_ranked=False)

        max_len = output_pad.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < torch.LongTensor(length_list).unsqueeze(1)).float()).to(self.device)
        output_pad = output_pad * mask.unsqueeze(-1)
        r_s = torch.div(torch.sum(output_pad, dim=1).transpose(0, 1),
                        torch.from_numpy(length_list).to(self.device)).transpose(0, 1)

        return r_s