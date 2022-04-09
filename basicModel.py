# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from DynamicRNN import DynamicRNN
from myutils import vectors2padsequence

class Lang:
    def __init__(self, vocab):
        self.index2word = {}
        self.word2index = {}
        for i in range(len(vocab)):
            self.index2word[i] = vocab[i]
            self.word2index[vocab[i]] = i

    def indexFromSentence(self, sentence, flag_list=True):
        list_ = sentence if flag_list else sentence.lower().split()
        list_idx = []
        for word in list_:
            list_idx.append(self.word2index[word] if word in self.word2index else self.word2index['<unk>'])
        return list_idx

    def VariablesFromSentences(self, sentences, flag_list=True, device=torch.device('cpu')):
        '''
        if sentence is a list of word, flag_list should be True in the training 
        '''
        indexes = [self.indexFromSentence(sen, flag_list) for sen in sentences]
        # torch.LongTensor(indexes)
        return Variable(torch.LongTensor(indexes)).to(device)

class CharLangModel(nn.Module):
    """docstring for CharLangModel"""
    def __init__(self, dim_char, dim_hidden_char, n_vocab_char, n_vocab, list_embed_char=None, n_layers=1, bias=True, batch_first=True, dropout=0, 
                        bidirectional=False, rnn_type='LSTM', device=torch.device('cpu')):
        super(CharLangModel, self).__init__()
        self.rnn_type = rnn_type

        self.add_module('embed_char', nn.Embedding(n_vocab_char, dim_char))
        self.add_module('dropout', nn.Dropout(dropout))
        self.add_module('rnn', DynamicRNN(dim_char, dim_hidden_char, n_layers, 
                                            bidirectional=bidirectional, rnn_type=rnn_type, device=device))
        self.add_module('linear_lm', nn.Linear(dim_hidden_char * (2 if bidirectional else 1), n_vocab, bias=False))

        if list_embed_char != None:
            self.embed_char.weight.data.copy_(torch.from_numpy(list_embed_char))

    def forward(self, input_char, length_word, length_sen, label_lm):
        e_char = self.embed_char(input_char)
        e_char = self.dropout(e_char)

        # encode word by characters
        output_pad_word, hidden_word = self.rnn(e_char, lengths=length_word, flag_ranked=False)

        h_out_word = hidden_word[0] if self.rnn_type == 'LSTM' else hidden_word
        v_word = torch.cat((h_out_word[-2], h_out_word[-1]), dim=1)

        # format the word encoded from RNN
        embedded_word_from_character = self.dropout(vectors2padsequence(v_word, length_sen))
        prob_log = F.log_softmax(self.linear_lm(embedded_word_from_character), dim=-1)

        loss_batch = []
        for p, g, l in zip(prob_log, label_lm, length_sen):
            loss_batch.append(F.nll_loss(p[: l], g[: l]))
        loss_char_lm = torch.mean(torch.stack(loss_batch))

        return embedded_word_from_character, loss_char_lm