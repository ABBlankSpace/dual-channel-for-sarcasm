# -*- coding: utf-8 -*-

import logging
from tqdm import tqdm
import json, argparse
import numpy as np
import pandas as pd
import re, math
from nltk import word_tokenize, pos_tag
from transformers import BertTokenizer
import spacy
nlp = spacy.load('en_core_web_sm')

def remove_url(text):
    results = re.compile(r'https://[a-zA-Z0-9.?/&=:]*', re.S)
    text = results.sub("", text)
    results = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
    text = results.sub("", text)
    return text

def remove_mention(text):
    return text
def remove_hashtag(text):
    return text

class DataManager(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        np.random.seed(FLAGS.seed)
        self.tokenizer = BertTokenizer.from_pretrained(self.FLAGS.bert_path, do_lower_case=True)

    # add mask for pos/neg
    def add_mask(self, content, senti_label):
        assert len(content) == len(senti_label)
        senti, nonsenti = [], []
        pos_num, neg_num = senti_label.count(1), senti_label.count(2)
        if pos_num >= neg_num: # consistent with literal sentiment and deep sentiment?
            for i, (token, senti_ele) in enumerate(zip(content, senti_label)):
                # if senti_ele == 1 or senti_ele == 2:
                #     senti.append(token)
                # else:
                #     nonsenti.append(token)
                if senti_ele == 0 or senti_ele == 1:
                    senti.append(token)
                else:
                    senti.append('[mask]')
                if senti_ele == 0 or senti_ele == 2:
                    nonsenti.append(token)
                else:
                    nonsenti.append('[mask]')
        else:
            for i, (token, senti_ele) in enumerate(zip(content, senti_label)):
                # if senti_ele == 2 or senti_ele == 1:
                #     senti.append(token)
                # else:
                #     nonsenti.append(token)
                if senti_ele == 0 or senti_ele == 2:
                    senti.append(token)
                else:
                    senti.append('[mask]')
                if senti_ele == 0 or senti_ele == 1:
                    nonsenti.append(token)
                else:
                    nonsenti.append('[mask]')
        senti = senti if len(senti) != 0 else content
        nonsenti = nonsenti if len(nonsenti) != 0 else content
        # if len(senti) == 0 or len(nonsenti) == 0:
        # print(senti, nonsenti)
        return senti, nonsenti
    # def add_mask(self, content, pos, neg, label):
    #     senti, nonsenti = [], []
    #     mask = []
    #     literal, deep = 1, 0
    #     if len(pos) + len(neg) == 0:
    #         senti, nonsenti = content, content
    #         if label == 0:
    #             literal, deep = 1, 1
    #         return senti, nonsenti#, literal, deep
    #     elif len(pos) >= len(neg):
    #         # if label == 0:
    #         #     mask, literal, deep = pos, 1, 1
    #         # else:
    #         #     mask, literal, deep = pos, 1, 0
    #         senti = pos
    #     else:
    #         # if label == 0:
    #         #     mask, literal, deep = neg, 0, 0
    #         # else:
    #         #     mask, literal, deep = neg, 0, 1
    #         senti = neg
    #     def ifinsenti(word, senti):
    #         all = []
    #         for ele in senti:
    #             if ele in word:
    #                 all.append(ele)
    #         return all
    #
    #     for ele in content:
    #         all = ifinsenti(ele, senti)
    #         if len(all) == 0:
    #         #     senti.extend(all)
    #         #     # nonsenti.append('[mask]')
    #         # else:
    #         #     # senti.append('[mask]')
    #             nonsenti.append(ele)
    #     # if len(pos) >= len(neg):
    #     #     senti, nonsenti = pos, nonsenti + neg
    #     # else:
    #     #     senti, nonsenti = neg, nonsenti + pos
    #     senti = senti if len(senti) != 0 else content
    #     nonsenti = nonsenti if len(nonsenti) != 0 else content
    #     # if len(senti) == 0 or len(nonsenti) == 0:
    #     # print(senti, nonsenti)
    #
    #     return senti, nonsenti#, literal, deep

    def add_position_encoding(self, senti_label):
        pos_posi, neg_posi = [], []
        for i in range(len(senti_label)):
            pos = []
            neg = []
            for j, ele in enumerate(senti_label):
                if ele == 1:
                    pos.append(abs(i - j))
                elif ele == 2:
                    neg.append(abs(i - j))
            pos = math.ceil(sum(pos) / len(pos)) if len(pos)>0 else 0
            neg = math.ceil(sum(neg) / len(neg)) if len(neg)>0 else 0
            pos_posi.append(pos)
            neg_posi.append(neg)
        return pos_posi, neg_posi

    def load_data(self, path, fname):
        f = open('%s%s' % (path, fname), encoding='utf8')
        lines = [json.loads(line.strip()) for line in f.readlines()]
        f.close()
        data = []
        for line in lines:
            origin, pos, neg, senti, nonsenti = line['content'].strip(), line['pos'], line['neg'], line['senti'], line['nonsenti']  # str, list
            senti = senti if len(senti) > 0 else nonsenti # pos+neg
            sarcasm = int(line['label'])
            senti_label = line['senti_label']
            # POS_label = line['POS_label']
            POS_label = []
            content = remove_url(origin).lower()
            if self.FLAGS.tokenizer == 'spacy':
                content = [str(token) for token in nlp(content)]
            elif self.FLAGS.tokenizer == 'nltk':
                content = word_tokenize(content)
            else:  # bert
                pass
            pos_posi, neg_posi = self.add_position_encoding(senti_label)

            if len(pos) >= len(neg) and sarcasm == 1:
                literal, deep = 1, 0
            # elif len(pos) == len(neg):
            #     literal, deep = 1, 1
            elif len(pos) < len(neg) and sarcasm == 1:
                literal, deep = 0, 1
            elif len(pos) >= len(neg) and sarcasm == 0:
                literal, deep = 1, 1
            elif len(pos) < len(neg) and sarcasm == 0:
                literal, deep = 0, 0
            pos_senti, pos_nonsenti, neg_senti, neg_nonsenti = [], [], [], []
            new_senti, new_nonsenti = [], [] # precise senti/nonsenti
            for (w, s, pp, nep) in zip(content, senti_label, pos_posi, neg_posi):
                if s == 1 or s == 2:
                    new_senti.append(w)
                    pos_senti.append(pp)
                    neg_senti.append(nep)
                else:
                    new_nonsenti.append(w)
                    pos_nonsenti.append(pp)
                    neg_nonsenti.append(nep)

            senti = senti if len(senti) > 0 else content  # (pos+neg)senti/nonsenti
            nonsenti = nonsenti if len(nonsenti) > 0 else content

            # senti, nonsenti = self.add_mask(content, senti_label) # pos/neg
            if len(content) > 0:
                oneline = {'content': content, 'sarcasm': sarcasm, 'senti': senti, 'nonsenti': nonsenti, 'origin': origin, 'literal': literal, 'deep': deep, 'senti_label': senti_label, 'POS_label': POS_label, 'pos_senti': pos_senti, 'pos_nonsenti': pos_nonsenti, 'neg_senti': neg_senti, 'neg_nonsenti': neg_nonsenti}
                data.append(oneline)
        np.random.shuffle(data)
        return data

    def gen_batched_data(self, data):
        max_len_sen = min(max([len(item['content']) for item in data]), self.FLAGS.max_length_sen)
        # print(max_len_sen)
        max_len_senti = min(max([len(item['senti']) for item in data]), self.FLAGS.max_length_sen)
        max_len_nonsenti = min(max([len(item['nonsenti']) for item in data]), self.FLAGS.max_length_sen)

        def padding(sent, l, pad='PAD'):
            return sent + [pad] * (l-len(sent))

        list_sentences, list_sentis, list_nonsentis, length_sen, length_senti, length_nonsenti = [], [], [], [], [], []
        literal, deep, sarcasm = [], [], []
        list_origin, list_masked_origin = [], []
        list_senti_labels = []
        list_pos_labels = []
        list_senti_pos, list_senti_neg = [], []
        list_nonsenti_pos, list_nonsenti_neg = [], []
        for item in data:
            sentence = item['content']
            sentence = sentence[:max_len_sen] if len(sentence) > max_len_sen else padding(sentence, max_len_sen)
            senti = item['senti']
            senti = senti[:max_len_senti] if len(senti) > max_len_senti else padding(senti, max_len_senti)
            nonsenti = item['nonsenti']
            senti_label = item['senti_label']
            senti_label = senti_label[:max_len_sen] if len(senti_label) > max_len_sen else (senti_label + [0]*(max_len_sen-len(senti_label)))
            POS_label = item['POS_label']
            POS_label = POS_label[:max_len_sen] if len(POS_label) > max_len_sen else padding(POS_label, max_len_sen, '.')
            assert len(sentence) == len(POS_label) == len(senti_label)
            list_pos_labels.append(POS_label)
            nonsenti = nonsenti[:max_len_nonsenti] if len(nonsenti) > max_len_nonsenti else padding(nonsenti, max_len_nonsenti)
            list_sentences.append(sentence)
            list_sentis.append(senti)
            list_nonsentis.append(nonsenti)
            length_sen.append(min(max_len_sen, len(item['content'])))
            length_senti.append(min(max_len_senti, len(item['senti'])))
            length_nonsenti.append(min(max_len_nonsenti, len(item['nonsenti'])))
            literal.append(item['literal'])
            deep.append(item['deep'])
            sarcasm.append(item['sarcasm'])
            list_senti_labels.append(senti_label)
            pos_senti = item['pos_senti'][:max_len_senti] if len(item['pos_senti']) > max_len_senti else padding(item['pos_senti'], max_len_senti)
            neg_senti = item['neg_senti'][:max_len_senti] if len(item['neg_senti']) > max_len_senti else padding(
                item['neg_senti'], max_len_senti)
            pos_nonsenti = item['pos_nonsenti'][:max_len_nonsenti] if len(item['pos_nonsenti']) > max_len_nonsenti else padding(
                item['pos_nonsenti'], max_len_nonsenti)
            neg_nonsenti = item['neg_nonsenti'][:max_len_nonsenti] if len(item['neg_nonsenti']) > max_len_nonsenti else padding(
                item['neg_nonsenti'], max_len_nonsenti)
            list_senti_pos.append(pos_senti)
            list_senti_neg.append(neg_senti)
            list_nonsenti_pos.append(pos_nonsenti)
            list_nonsenti_neg.append(neg_nonsenti)
            list_origin.append(item['origin'])
        batched_data = {'sentences': np.array(list_sentences),
                        'sentis': np.array(list_sentis),
                        'nonsentis': np.array(list_nonsentis),
                        'senti_labels': list_senti_labels,
                        'pos_labels': list_pos_labels,
                        'senti_neg': np.array(list_senti_neg),
                        'senti_pos': np.array(list_senti_pos),
                        'nonsenti_pos': np.array(list_nonsenti_pos),
                        'nonsenti_neg': np.array(list_nonsenti_neg),
                        'origins': list_origin,
                        # 'masked_origins': list_masked_origin,
                        'length_sen': np.array(length_sen),
                        'length_senti': np.array(length_senti),
                        'length_nonsenti': np.array(length_nonsenti),
                        'literals': np.array(literal),
                        'deeps': np.array(deep),
                        'sarcasms': np.array(sarcasm),
                        'max_len_sen': max_len_sen,
                        'max_len_senti': max_len_senti,
                        'max_len_nonsenti': max_len_nonsenti}
        return batched_data

    def load_data_bert(self, path, fname):
        f = open('%s%s' % (path, fname), encoding='utf8')
        lines = [json.loads(line.strip()) for line in f.readlines()]
        f.close()
        data = []
        num_literal_pos, num_literal_neg, num_literal_neu = 0, 0, 0
        num_deep_pos, num_deep_neg, num_deep_neu = 0, 0, 0
        for line in lines:
            origin, pos, neg, senti, nonsenti = line['content'].strip(), line['pos'], line['neg'], line[
                'senti'], line['nonsenti']  # str, list
            senti_label = line['senti_label']
            POS_label = line['POS_label']
            content = remove_url(origin).lower()
            senti = senti if len(senti) > 0 else nonsenti  # pos+neg
            sarcasm = int(line['label'])
            # if len(pos) > len(neg) and sarcasm == 1:
            #     literal, deep = 1, 2
            #     num_literal_pos += 1
            #     num_deep_neg += 1
            # # elif len(pos) == len(neg):
            # #     literal, deep = 0, 0
            # elif len(pos) == len(neg) and sarcasm == 1:
            #     literal, deep = 0, 2
            #     num_literal_neu += 1
            #     num_deep_neg += 1
            # elif len(pos) < len(neg) and sarcasm == 1:
            #     literal, deep = 2, 1
            #     num_literal_neg += 1
            #     num_deep_pos += 1
            # elif len(pos) > len(neg) and sarcasm == 0:
            #     literal, deep = 1, 1
            #     num_literal_pos += 1
            #     num_deep_pos += 1
            # elif len(pos) == len(neg) and sarcasm == 0:
            #     literal, deep = 0, 0
            #     num_literal_neu += 1
            #     num_deep_neu += 1
            # elif len(pos) < len(neg) and sarcasm == 0:
            #     literal, deep = 2, 2
            #     num_literal_neg += 1
            #     num_deep_neg += 1

            if len(pos) >= len(neg) and sarcasm == 1:
                literal, deep = 1, 0
                num_literal_pos += 1
                num_deep_neg += 1
            elif len(pos) < len(neg) and sarcasm == 1:
                literal, deep = 0, 1
                num_literal_neg += 1
                num_deep_pos += 1
            elif len(pos) >= len(neg) and sarcasm == 0:
                literal, deep = 1, 1
                num_literal_pos += 1
                num_deep_pos += 1
            elif len(pos) < len(neg) and sarcasm == 0:
                literal, deep = 0, 0
                num_literal_neg += 1
                num_deep_neg += 1
            senti = ' '.join(senti)
            nonsenti = ' '.join(nonsenti)
            content_tokenized_text = self.tokenizer.tokenize('[CLS] ' + content + ' [SEP]')
            senti_tokenized_text = self.tokenizer.tokenize('[CLS] ' + senti + ' [SEP]')
            nonsenti_tokenized_text = self.tokenizer.tokenize('[CLS] ' + nonsenti + ' [SEP]')
            assert len(content_tokenized_text) == len(senti_label)
            oneline = {'content': content_tokenized_text, 'senti': senti_tokenized_text, 'nonsenti': nonsenti_tokenized_text, 'sarcasm': sarcasm, 'origin': origin, 'literal': literal, 'deep': deep, 'senti_label': senti_label, 'POS_label': POS_label}
            # if len(line['senti']) > 0: # remove no senti
            if len(content_tokenized_text) > 0:
                # assert len(oneline['content']) == len(oneline['senti_label'])
                data.append(oneline)
        print('num_literal_pos: {}, num_literal_neg: {}, num_literal_neu: {}'.format(num_literal_pos, num_literal_neg, num_literal_neu))
        print('num_deep_pos: {}, num_deep_neg: {}, num_deep_neu: {}'.format(num_deep_pos, num_deep_neg,
                                                                                     num_deep_neu))
        np.random.shuffle(data)
        return data

    def gen_batched_data_bert(self, data):
        max_len_sen = min(max([len(item['content']) for item in data]), self.FLAGS.max_length_sen)
        max_len_senti = min(max([len(item['senti']) for item in data]), self.FLAGS.max_length_sen)
        # print(max_len_sen)
        max_len_nonsenti = min(max([len(item['nonsenti']) for item in data]), self.FLAGS.max_length_sen)

        def padding(sent, l, pad='PAD'):
            return sent + [pad] * (l-len(sent))

        list_sentences, list_sentis, list_nonsentis, length_sen, length_senti, length_nonsenti = [], [], [], [], [], []
        literal, deep, sarcasm = [], [], []
        list_senti_labels = []
        list_pos_labels = []
        for item in data:
            sentence = item['content']
            # assert len(sentence) == len(item['senti_label'])
            senti_label = item['senti_label'][:len(sentence)]
            sentence = self.tokenizer.convert_tokens_to_ids(sentence)
            senti = item['senti']
            senti = self.tokenizer.convert_tokens_to_ids(senti)
            nonsenti = item['nonsenti']
            nonsenti = self.tokenizer.convert_tokens_to_ids(nonsenti)
            if len(sentence) != len(senti_label):
                print(item['content'])
                print(senti_label)
            assert len(sentence) == len(senti_label)
            list_sentences.append(sentence)
            list_sentis.append(senti)
            list_nonsentis.append(nonsenti)
            list_senti_labels.append(senti_label)
            length_sen.append(min(max_len_sen, len(item['content'])))
            length_senti.append(min(max_len_senti, len(item['senti'])))
            length_nonsenti.append(min(max_len_nonsenti, len(item['nonsenti'])))
            literal.append(item['literal'])
            deep.append(item['deep'])
            sarcasm.append(item['sarcasm'])
        batched_data = {'sentences': np.array(list_sentences),
                        'sentis': np.array(list_sentis),
                        'nonsentis': np.array(list_nonsentis),
                        'senti_labels': list_senti_labels,
                        # 'origins': list_origin,
                        # 'masked_origins': list_masked_origin,
                        'length_sen': np.array(length_sen),
                        'length_senti': np.array(length_senti),
                        'length_nonsenti': np.array(length_nonsenti),
                        'literals': np.array(literal),
                        'deeps': np.array(deep),
                        'sarcasms': np.array(sarcasm),
                        # 'origins': list_origin,
                        'max_len_sen': max_len_sen,
                        'max_len_senti': max_len_senti,
                        'max_len_nonsenti': max_len_nonsenti}
        return batched_data

    def load_data_UCDCC(self, path, fname, cut=3):
        f = open('%s%s' % (path, fname), encoding='utf8')
        lines = [json.loads(line.strip()) for line in f.readlines()]
        f.close()
        data = []
        # double dataset by cut
        for line in lines:
            origin, pos, neg, senti, nonsenti = line['content'].strip(), line['pos'], line['neg'], line['senti'], line['nonsenti']  # str, list
            senti = senti if len(senti) > 0 else nonsenti # pos+neg
            sarcasm = int(line['label'])
            senti_label = line['senti_label']
            POS_label = line['POS_label']
            content = word_tokenize(remove_url(origin).lower())

            if len(pos) >= len(neg) and sarcasm == 1:
                literal, deep = 1, 0
            # elif len(pos) == len(neg):
            #     literal, deep = 0, 0
            elif len(pos) < len(neg) and sarcasm == 1:
                literal, deep = 0, 1
            elif len(pos) >= len(neg) and sarcasm == 0:
                literal, deep = 1, 1
            elif len(pos) < len(neg) and sarcasm == 0:
                literal, deep = 0, 0
            # senti = senti if len(senti) > 0 else content  # pos+neg
            # nonsenti = nonsenti if len(nonsenti) > 0 else content  # pos+neg
            if len(content) > 0:
                if len(content) > cut:
                    senti, nonsenti = content[:cut], content[cut:]
                    oneline = {'content': content, 'sarcasm': sarcasm, 'senti': senti, 'nonsenti': nonsenti,
                               'origin': origin, 'literal': literal, 'deep': deep, 'senti_label': senti_label, 'POS_label': POS_label}
                    data.append(oneline)

                    senti, nonsenti = content[:-cut], content[-cut:]
                    oneline = {'content': content, 'sarcasm': sarcasm, 'senti': senti, 'nonsenti': nonsenti, 'origin': origin, 'literal': literal, 'deep': deep, 'senti_label': senti_label, 'POS_label': POS_label}
                    data.append(oneline)
                else:
                    senti, nonsenti = content, content
                    oneline = {'content': content, 'sarcasm': sarcasm, 'senti': senti, 'nonsenti': nonsenti,
                               'origin': origin, 'literal': literal, 'deep': deep, 'senti_label': senti_label, 'POS_label': POS_label}
                    data.append(oneline)
        np.random.shuffle(data)
        return data

    def build_vocab(self, path, data, vocab=dict()):
        logging.info('Building vocabulary...')
        pos = dict()
        for inst in tqdm(data):
            try:
                for word in inst['POS_label']:
                    pos[word] = (pos[word]+1) if (word in pos) else 1
            except:
                break
        pos_list = sorted(pos, key=pos.get, reverse=True)

        for inst in tqdm(data):
            for word in inst['content']:
                vocab[word] = (vocab[word]+1) if (word in vocab) else 1
        vocab_list = sorted(vocab, key=vocab.get, reverse=True)

        vocab_list = vocab_list[: min(len(vocab_list), self.FLAGS.voc_size)]
        if '<unk>' not in vocab:
            vocab_list.append('<unk>')
        if '[mask]' not in vocab:
            vocab_list.append('[mask]')
        # save vocab list
        # print(len(vocab_list))
        f = open(self.FLAGS.data_dir + 'vocab.txt', 'w', encoding='utf8')
        for word in vocab_list:
            f.write(word + '\n')
        f.close()

        vocab_wordvec = dict()
        logging.info('Loading word vectors...')
        vectors = {}
        with open('./senti/glove.840B.300d.txt', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i % 20000 == 0:
                    logging.info("processing line %d" % i)
                s = line.strip()

                word = s[:s.find(' ')]
                vector = s[s.find(' ') + 1:]
                vectors[word] = vector

        f = open(self.FLAGS.data_dir + 'vectors.glove.300d.txt', 'w', encoding='utf8')
        embed = []
        num_not_found, num_found = 0, 0
        for word in vocab_list:
            if word in vectors:
                vector = list(map(float, vectors[word].split()))
                f.write(word + ' ' + vectors[word] + '\n') # only in
                num_found += 1
                vocab_wordvec[word] = None  ###
            else:
                num_not_found += 1
                vector = np.random.random(self.FLAGS.dim_input) * 0.1
            embed.append(vector) # all
            # print(vector)
        logging.info('%s words found in vocab' % num_found)
        logging.info('%s words not found in vocab' % num_not_found)
        embed = np.array(embed, dtype=np.float32)

        f.close()
        # return POS vocab
        return vocab_list, embed, vocab_wordvec, pos_list

    def load_vocab_embed(self):
        logging.info('Loading existed vocabulary...')
        vocab_list = []
        f = open(self.FLAGS.data_dir + 'vocab.txt', 'r', encoding='utf8')
        for line in f.readlines():
            vocab_list.append(line.strip())
        f.close()

        vocab_wordvec = dict()
        logging.info('Loading word vectors...')
        vectors = {}
        with open(self.FLAGS.data_dir + 'vectors.glove.300d.txt', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i % 20000 == 0:
                    logging.info("processing line %d" % i)
                s = line.strip()

                word = s[:s.find(' ')]
                vector = s[s.find(' ') + 1:]
                vectors[word] = vector

        embed = []
        num_not_found, num_found = 0, 0
        for word in vocab_list:
            if word in vectors:
                vector = list(map(float, vectors[word].split()))
                num_found += 1
            else:
                num_not_found += 1
                vector = np.random.random(self.FLAGS.dim_word) * 0.1
            embed.append(vector)  # all
        logging.info('%s words found in vocab' % num_found)
        logging.info('%s words not found in vocab' % num_not_found)
        embed = np.array(embed, dtype=np.float32)

        return vocab_list, embed


# parser = argparse.ArgumentParser()
# parser.add_argument('--voc_size', type=int, default=32768)
# parser.add_argument('--dim_word', type=int, default=300, choices=[100, 200, 300])
# FLAGS = parser.parse_args()

# data_dir = 'senti/'
# path_wordvec = '../SentimentInDialog/dailydialog/glove.840B.300d.txt'
# dm = DataManager(FLAGS)
# dataset_name = ('train', 'test', 'valid')
# data = {}
# for tmp in dataset_name:
#     data[tmp] = dm.load_data_senti(data_dir, f'{tmp}.csv') #
#
# vocab, embed, vocab_dict = dm.build_vocab('%s' % (path_wordvec), data['train']+data['test']+data['valid'])
