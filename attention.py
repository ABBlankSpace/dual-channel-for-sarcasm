# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from myutils import dynamic_softmax

class Attention(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, device=torch.device('cpu')):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(Attention, self).__init__()
        self.attention_size = attention_size
        self.device = device
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.attention_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({attention_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        logits = inputs.matmul(self.attention_vector) # all_utter * max_sen_len * dim_sen dim_sen=all_utter * max_sen_len
        unnorm_ai = (logits - logits.max()).exp() # all_utter * max_sen_len

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float())
        mask = mask.to(self.device) if self.device else mask

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums) # scalar utter*max_sen_len

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs)) # utter*max_sen_len*dim_sen utter*max_sen_len*dim_sen=utter*max_sen_len*dim_sen

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1) # utter*dim_sen

        return representations, attentions

class Attention_Modified(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, dim_profile, dim_sen, device=torch.device('cpu')):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(Attention_Modified, self).__init__()
        self.attention_size = attention_size
        self.device = device
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))

        self.w_q = nn.Linear(dim_profile, dim_sen)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.attention_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({attention_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths, aspects):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        # dot product attention
        # inputs: utter*max_sen_len*dim_sen
        # aspects: utter*max_sen_len*[2*]dim_profile # user+intent
        aspects_x = self.w_q(aspects) # utter*max_sen_len*dim_sen
        inputs_t = inputs.permute(0, 2, 1)
        logits = torch.bmm(aspects_x, inputs_t) # utter*max_sen_len*max_sen_len
        score = F.softmax(logits, dim=-1)
        weighted = torch.bmm(score, inputs)

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return representations, score


class AttentionPair(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, dim_vect, dim_attn, flag_bid, device=torch.device('cpu')):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(AttentionPair, self).__init__()
        self.device = device
        dim_attn_bid = dim_attn * (2 if flag_bid else 1)
        self.add_module('linear_vec', nn.Linear(dim_vect, dim_attn, bias=False))
        self.add_module('linear_mat', nn.Linear(dim_attn_bid, dim_attn, bias=False))
        self.add_module('linear_attn', nn.Linear(dim_attn, 1, bias=False))

    def forward(self, vector, matrix, input_lengths):
        """ Forward pass.
        # Arguments:
            vect (Torch.Variable): Tensor of input vector
            matrix (Torch.Variable): Tensor of input matrix
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        t1 = self.linear_vec(vector)
        t2 = self.linear_mat(matrix)
        t3 = F.relu(t1.unsqueeze(1) + t2)
        logits = self.linear_attn(t3).squeeze(-1)
        unnorm_ai = (logits - logits.max()).exp()

        attentions = dynamic_softmax(unnorm_ai, torch.LongTensor(input_lengths), self.device)

        # apply attention weights
        weighted = torch.mul(matrix, attentions.unsqueeze(-1).expand_as(matrix))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return representations, attentions

class AttentionTri(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, dim_vect, dim_attn, flag_bid, device=torch.device('cpu')):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(AttentionTri, self).__init__()
        self.device = device
        dim_attn_bid = dim_attn * (2 if flag_bid else 1)
        self.add_module('linear_vec', nn.Linear(dim_vect, dim_attn, bias=False))
        self.add_module('linear_vec2', nn.Linear(dim_vect, dim_attn, bias=False))
        self.add_module('linear_mat', nn.Linear(dim_attn_bid, dim_attn, bias=False))
        self.add_module('linear_attn', nn.Linear(dim_attn, 1, bias=False))

    def forward(self, vector, vector2, matrix, input_lengths):
        """ Forward pass.
        # Arguments:
            vect (Torch.Variable): Tensor of input vector
            matrix (Torch.Variable): Tensor of input matrix
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        t1 = self.linear_vec(vector) + self.linear_vec2(vector2)
        t2 = self.linear_mat(matrix)
        t3 = F.relu(t1 + t2)
        logits = self.linear_attn(t3).squeeze(-1)
        unnorm_ai = (logits - logits.max()).exp()

        attentions = dynamic_softmax(unnorm_ai, torch.LongTensor(input_lengths), self.device)

        # apply attention weights
        weighted = torch.mul(matrix, attentions.unsqueeze(-1).expand_as(matrix))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return representations, attentions