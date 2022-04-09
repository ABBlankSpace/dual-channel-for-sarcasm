# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from basicModel import Lang
from dual_birnn import dualModel

class bridgeModel(nn.Module):
    def __init__(self, FLAGS, vocab=None, embed=None):
        super(bridgeModel, self).__init__()
        self.device = FLAGS.device
        self.max_length_sen = FLAGS.max_length_sen
        self.n_class = FLAGS.n_class
        self.learning_rate = FLAGS.learning_rate
        self.batch_size = FLAGS.batch_size
        self.lambda1 = FLAGS.lambda1
        self.bidirectional = True if FLAGS.bidirectional else False
        print('****Bidirectional in bridgeModel is {}'.format(self.bidirectional))
        self.margin = FLAGS.margin
        self.supervised = True if FLAGS.supcon else False
        self.temp = FLAGS.temp
        self.t_sne = FLAGS.t_sne

        self.lang = Lang(vocab)
        posi = [i for i in range(self.max_length_sen-1)]
        posi.append('<unk>')
        self.lang_posi = Lang(posi)
        self.model = dualModel(FLAGS, len(vocab), embed)

        self.model.to(self.device)

        self.optimizer = getattr(optim, FLAGS.optim_type)([
            {'params': self.model.base_params, 'weight_decay': FLAGS.weight_decay},
            {'params': self.model.sen.embed.parameters(), 'lr': FLAGS.lr_word_vector, 'weight_decay':0}, # fine tuning,
            {'params': self.model.literal.embed.parameters(), 'lr': 0, 'weight_decay':0},
            {'params': self.model.deep.embed.parameters(), 'lr': 0, 'weight_decay': 0}],
            lr=self.learning_rate)

    def gen_batch_data(self, batched_data):
        dict_data = {}
        dict_data['sens'] = self.lang.VariablesFromSentences(batched_data['sentences'], True, self.device)
        dict_data['len_sen'] = batched_data['length_sen']
        # dict_data['POS'] = batched_data['pos_labels']

        # dict_data['senti_pos'] = self.lang_posi.VariablesFromSentences(batched_data['senti_pos'], True, self.device)
        # dict_data['senti_neg'] = self.lang_posi.VariablesFromSentences(batched_data['senti_neg'], True, self.device)
        # dict_data['nonsenti_pos'] = self.lang_posi.VariablesFromSentences(batched_data['nonsenti_pos'], True, self.device)
        # dict_data['nonsenti_neg'] = self.lang_posi.VariablesFromSentences(batched_data['nonsenti_neg'], True, self.device)

        dict_data['senti_sens'] = self.lang.VariablesFromSentences(batched_data['sentis'], True, self.device)
        dict_data['senti_len_sen'] = batched_data['length_senti']

        dict_data['nonsenti_sens'] = self.lang.VariablesFromSentences(batched_data['nonsentis'], True, self.device)
        dict_data['nonsenti_len_sen'] = batched_data['length_nonsenti']
        literals = Variable(torch.LongTensor(batched_data['literals']))
        dict_data['literals'] = literals.to(self.device) if self.device else literals
        deeps = Variable(torch.LongTensor(batched_data['deeps']))
        dict_data['deeps'] = deeps.to(self.device) if self.device else deeps
        sarcasms = Variable(torch.LongTensor(batched_data['sarcasms']))
        dict_data['sarcasms'] = sarcasms.to(self.device) if self.device else sarcasms
        return dict_data

    def predict(self, batched_data):
        self.model.eval()
        b_data = self.gen_batch_data(batched_data) # padded idx
        with torch.no_grad():
            if self.t_sne:
                prob, _, _, literal_rep, deep_rep, sen_rep = self.model(b_data)
            else:
                prob, _, _ = self.model(b_data)
        label_idx = [tmp.item() for tmp in torch.argmax(prob, dim=-1)]
        if self.t_sne:
            return label_idx, literal_rep, deep_rep, sen_rep
        else:
            return label_idx

    def stepTrain(self, batched_data, inference=False):
        self.model.eval() if inference else self.model.train()

        if inference == False:
            self.optimizer.zero_grad()

        b_data = self.gen_batch_data(batched_data)

        prob, prob_senti, prob_nonsenti = self.model(b_data)
        # prob, prob_senti, prob_nonsenti, literal, deep, sen = self.model(b_data)
        loss_ce = F.nll_loss(torch.log(prob), b_data['sarcasms'])
        loss_senti = F.nll_loss(torch.log(prob_senti), b_data['literals'])
        loss_nonsenti = F.nll_loss(torch.log(prob_nonsenti), b_data['deeps'])

        loss_all = loss_ce + loss_senti + loss_nonsenti

        if inference == False:
            loss_all.backward()
            self.optimizer.step()

        return np.array([loss_all.data.cpu().numpy(), loss_ce.data.cpu().numpy(), loss_all.data.cpu().numpy(),
                         loss_all.data.cpu().numpy()]).reshape(
            4), prob.data.cpu().numpy(), prob_senti.data.cpu().numpy(), prob_nonsenti.data.cpu().numpy()

    def save_model(self, dir, idx):
        os.mkdir(dir) if not os.path.isdir(dir) else None
        torch.save(self.state_dict(), '%s/model%s.pth' % (dir, idx))
        # torch.save(self, '%s/model%s.pkl' % (dir, idx))
        # print all params
        print('****save state dict****')
        print(self.state_dict().keys())


    def load_model(self, dir, idx=-1, device='cpu'):
        if idx < 0:
            params = torch.load('%s' % dir)
            self.load_state_dict(params)
        else:
            print('****load state dict****')
            print(self.state_dict().keys())
            self.load_state_dict(torch.load(f'{dir}/model{idx}.pth', map_location=device))
