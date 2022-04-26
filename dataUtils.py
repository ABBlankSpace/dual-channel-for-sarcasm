# -*- coding: utf-8 -*-

import logging
from tqdm import tqdm
import json
import numpy as np
import re
from nltk import word_tokenize
from transformers import BertTokenizer
import spacy
nlp = spacy.load('en_core_web_sm')

def remove_url(text):
    results = re.compile(r'https://[a-zA-Z0-9.?/&=:]*', re.S)
    text = results.sub("", text)
    results = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
    text = results.sub("", text)
    return text

class DataManager(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        np.random.seed(FLAGS.seed)
        self.tokenizer = BertTokenizer.from_pretrained(self.FLAGS.bert_path, do_lower_case=True)

    def load_data(self, path, fname):
        f = open('%s%s' % (path, fname), encoding='utf8')
        lines = [json.loads(line.strip()) for line in f.readlines()]
        f.close()
        data = []
        for line in lines:
            origin, pos, neg, senti, nonsenti = line['content'].strip(), line['pos'], line['neg'], line['senti'], line['nonsenti']  # str, list
            senti = senti if len(senti) > 0 else nonsenti # pos+neg
            sarcasm = int(line['label'])
            content = remove_url(origin).lower()
            if self.FLAGS.tokenizer == 'spacy':
                content = [str(token) for token in nlp(content)]
            elif self.FLAGS.tokenizer == 'nltk':
                content = word_tokenize(content)
            else:  # bert
                pass

            if len(pos) >= len(neg) and sarcasm == 1:
                literal, deep = 1, 0
            elif len(pos) < len(neg) and sarcasm == 1:
                literal, deep = 0, 1
            elif len(pos) >= len(neg) and sarcasm == 0:
                literal, deep = 1, 1
            elif len(pos) < len(neg) and sarcasm == 0:
                literal, deep = 0, 0
            else:
                literal, deep = 0, 0

            senti = senti if len(senti) > 0 else content
            nonsenti = nonsenti if len(nonsenti) > 0 else content

            if len(content) > 0:
                oneline = {'content': content, 'sarcasm': sarcasm, 'senti': senti, 'nonsenti': nonsenti, 'origin': origin, 'literal': literal, 'deep': deep}
                data.append(oneline)
        np.random.shuffle(data)
        return data

    def gen_batched_data(self, data):
        max_len_sen = min(max([len(item['content']) for item in data]), self.FLAGS.max_length_sen)
        max_len_senti = min(max([len(item['senti']) for item in data]), self.FLAGS.max_length_sen)
        max_len_nonsenti = min(max([len(item['nonsenti']) for item in data]), self.FLAGS.max_length_sen)

        def padding(sent, l, pad='PAD'):
            return sent + [pad] * (l-len(sent))

        list_sentences, list_sentis, list_nonsentis, length_sen, length_senti, length_nonsenti = [], [], [], [], [], []
        literal, deep, sarcasm = [], [], []
        list_origin = []
        for item in data:
            sentence = item['content']
            sentence = sentence[:max_len_sen] if len(sentence) > max_len_sen else padding(sentence, max_len_sen)
            senti = item['senti']
            senti = senti[:max_len_senti] if len(senti) > max_len_senti else padding(senti, max_len_senti)
            nonsenti = item['nonsenti']
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
            list_origin.append(item['origin'])
        batched_data = {'sentences': np.array(list_sentences),
                        'sentis': np.array(list_sentis),
                        'nonsentis': np.array(list_nonsentis),
                        'origins': list_origin,
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

    def build_vocab(self, path, data, vocab=dict()):
        logging.info('Building vocabulary...')
        for inst in tqdm(data):
            for word in inst['content']:
                vocab[word] = (vocab[word]+1) if (word in vocab) else 1
        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        vocab_list = vocab_list[: min(len(vocab_list), self.FLAGS.voc_size)]
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
                f.write(word + ' ' + vectors[word] + '\n')
                num_found += 1
                vocab_wordvec[word] = None
            else:
                num_not_found += 1
                vector = np.random.random(self.FLAGS.dim_input) * 0.1
            embed.append(vector)
        logging.info('%s words found in vocab' % num_found)
        logging.info('%s words not found in vocab' % num_not_found)
        embed = np.array(embed, dtype=np.float32)

        f.close()
        return vocab_list, embed, vocab_wordvec
