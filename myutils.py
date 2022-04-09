# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import re, nltk, requests, json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

# from simhash import Simhash

def vectors2padsequence(vectors, lengths):
    embedded_ = []
    idx_begin, idx_end = 0, 0
    for len_current in lengths:
        idx_begin, idx_end = idx_end, idx_end + len_current
        embedded_tmp = vectors[idx_begin: idx_end]
        embedded_.append(embedded_tmp)

    # rank the embedded firstly
    idx_ = np.argsort(lengths)[::-1]
    embedded_ranked = [embedded_[i] for i in idx_]
    _embedded = pad_sequence(embedded_ranked, batch_first=True)

    embedded_recover = torch.stack([_embedded[i, ...] for i in np.argsort(idx_)])
    return embedded_recover

def vectors2padsequence_modified(vectors, lengths, max_len):
    embedded_ = []
    idx_begin, idx_end = 0, 0
    for len_current in lengths:
        idx_begin, idx_end = idx_end, idx_end + len_current
        embedded_tmp = vectors[idx_begin: idx_end]
        embedded_.append(embedded_tmp)

    # rank the embedded firstly
    idx_ = np.argsort(lengths)[::-1]
    embedded_ranked = [embedded_[i] for i in idx_]
    _embedded = pad_sequence_modified(embedded_ranked, max_len, batch_first=True)

    embedded_recover = torch.stack([_embedded[i, ...] for i in np.argsort(idx_)])
    return embedded_recover

def pad_sequence_modified(sequences, max_len, batch_first=False, padding_value=0):
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    # max_len = max([s.size(0) for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

def dynamic_softmax(input, input_lengths, use_cuda=False):
    """ Forward pass.
    # Arguments:
        inputs (Torch.Variable): Tensor of input matrix
        input_lengths (torch.LongTensor): Lengths of the effective each row
    # Return:
        attentions: dynamic softmax results
    """
    mask = mask_gen(input_lengths, use_cuda)

    # apply mask and renormalize attention scores (weights)
    masked_weights = input * mask
    att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
    dyn_softmax = masked_weights.div(att_sums)

    return dyn_softmax


def mask_gen(input_lengths, device=torch.device('cpu')):
    """ Forward pass.
    # Arguments:
        input_lengths (torch.LongTensor): Lengths of the effective each row
    # Return:
        mask: mask results
    """
    max_len = torch.max(input_lengths)
    indices = torch.arange(0, max_len).unsqueeze(0)
    mask = Variable((indices < input_lengths.unsqueeze(1)).float()).to(device)

    return mask

def GenMeanVectorForText(embed, lengths, device=torch.device('cpu')):
    variable_len = Variable(torch.FloatTensor(1.0/lengths)).unsqueeze(-1).to(device)
    mask = mask_gen(torch.LongTensor(lengths), device)
    embed_masked = embed * mask.unsqueeze(-1)
    v_text = torch.sum(embed_masked, 1) * variable_len

    return v_text

def cleanText(text):
    def add_space(matched):
        s = matched.group()
        return ' '+ s[0] + ' ' + s[-1]
    
    con_cleaned = re.sub(r'[^a-zA-Z0-9_\-\.,;:!?/\']', " ", text)
    con_cleaned = re.sub(r'[\.,;:!?/]+[a-zA-Z]', add_space, con_cleaned)
    
    try:
        wordtoken = nltk.word_tokenize(con_cleaned)
    except:
        print(con_cleaned)
        print(text)
        exit()
    content_tackled = ' '.join(wordtoken)

    def add_space_pre(matched):
        '''
        If word like "china." occured, split "china" and ".". 
        '''
        s = matched.group()
        return s[0] + ' ' + s[-1]
    content_tackled = re.sub(r'[a-zA-Z][\.,;:!?/]+', add_space_pre, content_tackled)
    def remove_space_pre(matched):
        s = matched.group()
        return s[1:]
    # content_tackled = re.sub(r' [\'\.,;:!?/]+[srmdt]', remove_space_pre, content_tackled)
    
    return content_tackled

def segbot(text, port=9601):
    key = "AE9"
    url_segbot = 'http://115.182.62.169:%s/segbot' % port
    # url_segbot = 'http://155.69.151.69:%s/segbot' % port
    headers = {'Content-Type': 'application/json'}
    params = {'key': key, 'text': cleanText(text), 'level': 'paragraph'}
    try:
        response = requests.post(url=url_segbot, headers=headers, data=json.dumps(params).encode('utf-8'))
    except:
        try:
            response = requests.post(url=url_segbot, headers=headers, data=json.dumps(params).encode('utf-8'))
        except:
            pass
    return eval(response.text)['seg_result']


# def text2simhash(text):
#     def get_features(s):
#         width = 3
#         s = s.lower()
#         s = re.sub(r'[^\w]+', '', s)
#         return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
#     if isinstance(text, list):
#         text = ' '.join(text)
#     return Simhash(get_features(text)).value