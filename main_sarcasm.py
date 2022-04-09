# -*- coding: utf-8 -*-

from data_utils import DataManager
from tensorboard_logger import configure, log_value
import numpy as np
import random, logging, argparse, os, time, json
import torch
from ChatEvaluation import evaluateClassification, evaluateClassification_sarcasm
from bridgeModel_sarcasm import bridgeModel
# from tsne import t_sne_plot

parser = argparse.ArgumentParser()
parser.add_argument('--voc_size', type=int, default=30000) # 4500
parser.add_argument('--dim_input', type=int, default=300, choices=[100, 200, 300])
parser.add_argument('--dim_hidden', type=int, default=256, choices=[128, 256, 300, 512, 768])
parser.add_argument('--dim_bert', type=int, default=768)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_class', type=int, default=2, choices=[2, 6, 7])
parser.add_argument('--bidirectional', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lr_word_vector', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--embed_dropout_rate', type=float, default=0.5)
parser.add_argument('--cell_dropout_rate', type=float, default=0.5) # use in lstm multi-layer
parser.add_argument('--final_dropout_rate', type=float, default=0.5) # use?
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_length_sen', type=int, default=100) # 32
parser.add_argument('--iter_num', type=int, default=32*150)
parser.add_argument('--per_checkpoint', type=int, default=16) #
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
parser.add_argument('--data_dir', type=str, default='./debatev2/spacy/')
parser.add_argument('--breakpoint', type=int, default=56)
parser.add_argument('--path_wordvec', type=str, default='glove.840B.300d.txt')
parser.add_argument('--name_model', type=str, default='BertAtt')
parser.add_argument('--name_dataset', type=str, default='match')
parser.add_argument('--bert_path', type=str, default='../SentimentInDialog/uncased_L-12_H-768_A-12/')
parser.add_argument('--bert_config_path', type=str, default='../SentimentInDialog/uncased_L-12_H-768_A-12/config.json')
parser.add_argument('--bert_pretrained_path', type=str, default='../SentimentInDialog/uncased_L-12_H-768_A-12/pytorch_model.bin')
parser.add_argument('--pretrained_model_path', type=str, default='../SentimentInDialog/uncased_L-12_H-768_A-12/pytorch_model.bin')
parser.add_argument('--linear_dropout_rate', type=float, default=0.1)
parser.add_argument('--lr_bert', type=float, default=5e-05)
parser.add_argument('--predict_dir', type=str, default='./senti/')
parser.add_argument('--predict', type=int, default=1)
parser.add_argument('--model_dir', type=str, default='./models/debatev2dualbilstm-v5-c-ce-1-1-pc-32/')
parser.add_argument('--multi_dim', type=int, default=20)
parser.add_argument('--cut', type=int, default=3)
parser.add_argument('--margin', type=int, default=2)
parser.add_argument('--temp', type=float, default=0.07) # temperature for supervised contrast learning
parser.add_argument('--supcon', type=int, default=0)
parser.add_argument('--split_rate', type=float, default=0.5)
parser.add_argument('--tokenizer', type=str, default='spacy') # nltk, spacy, bert

parser.add_argument('--d_trans', type=int, default=2048)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_trans_layer', type=int, default=4)
parser.add_argument('--trans_dropout_rate', type=float, default=0.1)
parser.add_argument('--t_sne', type=int, default=1) # together with --predict
parser.add_argument('--save_model', type=int, default=0)
FLAGS = parser.parse_args()
print(FLAGS)

np.random.seed(FLAGS.seed)
random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.cuda.manual_seed(FLAGS.seed)
    FLAGS.device = torch.device('cuda') # not a specific one
else:
    FLAGS.device = torch.device('cpu')

def train(model ,datamanager, data_train, sample=False):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = datamanager.gen_batched_data(selected_data)
    loss, _, _, _ = model.stepTrain(batched_data)
    return loss

def predict(model, datamanager, data_):
    st, ed, times = 0, FLAGS.batch_size, 0
    y_true, y_pred = [], []
    cont = []
    # if FLAGS.t_sne:
    liter, deep, sen, idx, sentis, nonsentis = [], [], [], [], [], []
    while st < min(len(data_), 1e4):
        selected_data = data_[st:ed]
        batched_data = datamanager.gen_batched_data(selected_data)
        cont.extend(batched_data['origins'])
        x = np.arange(st, ed)
        idx.extend(x)
        sentis.extend(batched_data['sentis'])
        nonsentis.extend(batched_data['nonsentis'])
        if FLAGS.t_sne:
            label_idx, liter_rep, deep_rep, sen_rep = model.predict(batched_data)
            # type of literal_rep is torch.Tensor, turn into numpy
            # detect type?
            liter.extend(liter_rep.cpu().numpy())
            deep.extend(deep_rep.cpu().numpy())
            sen.extend(sen_rep.cpu().numpy())
        else:
            label_idx = model.predict(batched_data)
        y_true.extend(batched_data['sarcasms'])
        y_pred.extend(label_idx)
        st, ed = ed, ed + FLAGS.batch_size
    # t_sne
    if FLAGS.t_sne:

        liter, deep, sen = np.array(liter), np.array(deep), np.array(sen) # save representations, embedding/word vector
        # t_sne_data = np.vstack((liter, deep))
        # t_sne_data = sen
        t_sne_label = np.array(y_true)
        # t_sne_plot(t_sne_data, t_sne_label)
        # only save right ones
        sentis, nonsentis, cont, idx = np.array(sentis), np.array(nonsentis), np.array(cont), np.array(idx)
        np.save('tsne/'+FLAGS.name_dataset+'-'+FLAGS.name_model+'-'+str(FLAGS.breakpoint)+'-liter.npy', liter)
        np.save('tsne/'+FLAGS.name_dataset + '-' + FLAGS.name_model + '-' + str(FLAGS.breakpoint) + '-deep.npy', deep)
        np.save('tsne/'+FLAGS.name_dataset + '-' + FLAGS.name_model + '-' + str(FLAGS.breakpoint) + '-sen.npy', sen)
        np.save('tsne/'+FLAGS.name_dataset + '-' + FLAGS.name_model + '-' + str(FLAGS.breakpoint) + '-sarc.npy', t_sne_label)
        np.save('tsne/'+FLAGS.name_dataset + '-' + FLAGS.name_model + '-' + str(FLAGS.breakpoint) + '-pred.npy', np.array(y_pred))
        np.save('tsne/' + FLAGS.name_dataset + '-' + FLAGS.name_model + '-' + str(FLAGS.breakpoint) + '-senti.npy',
                sentis)
        np.save('tsne/' + FLAGS.name_dataset + '-' + FLAGS.name_model + '-' + str(FLAGS.breakpoint) + '-nonsenti.npy',
                nonsentis)
        np.save('tsne/' + FLAGS.name_dataset + '-' + FLAGS.name_model + '-' + str(FLAGS.breakpoint) + '-cont.npy',
                cont)
        np.save('tsne/' + FLAGS.name_dataset + '-' + FLAGS.name_model + '-' + str(FLAGS.breakpoint) + '-idx.npy',
                idx)

    write_res(cont.tolist(), idx.tolist(), sentis.tolist(), nonsentis.tolist(), y_true, y_pred)

def write_res(content, idx, sentis, nonsentis, sarcasm_label, senti_label):
    assert len(content) == len(sarcasm_label) == len(senti_label)
    f = open(FLAGS.data_dir + 'test_res.txt', 'w', encoding='utf8')
    for i in range(len(content)):
        oneline = {'idx': int(idx[i]),
                   'content': content[i].strip(),
                   'sentis': sentis[i].tolist(),
                   'nonsentis': nonsentis[i].tolist(),
                    'true': int(sarcasm_label[i]),
                    'pred': int(senti_label[i])}
        # print(oneline)
        f.write(json.dumps(oneline, ensure_ascii=False) + '\n')
    f.close()

num_loss = 4 ###

def evaluate(model, datamanager, data_):
    loss = np.zeros((num_loss, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    y_pred_literal, y_true_literal = [], []
    y_pred_deep, y_true_deep = [], []
    y_pred_sarcasm, y_true_sarcasm = [], []
    while st < min(len(data_), 1e4): # 10000
        selected_data = data_[st:ed]
        batched_data = datamanager.gen_batched_data(selected_data)
        _loss, pro_sarcasm, pro_literal, pro_deep = model.stepTrain(batched_data, inference=True)  # inference control model.train()/model.eval()
        y_true_literal.extend(batched_data['literals'])
        y_true_deep.extend(batched_data['deeps'])
        y_true_sarcasm.extend(batched_data['sarcasms'])
        y_pred_sarcasm.extend(pro_sarcasm)
        y_pred_literal.extend(pro_literal)
        y_pred_deep.extend(pro_deep)
        loss += _loss
        st, ed = ed, ed + FLAGS.batch_size
        times += 1
    loss /= times

    dict_evaluation = dict()
    dict_evaluation['literal'] = evaluateClassification(y_true_literal, y_pred_literal)
    dict_evaluation['deep'] = evaluateClassification(y_true_deep, y_pred_deep)
    dict_evaluation['sarcasm'] = evaluateClassification(y_true_sarcasm, y_pred_sarcasm)
    # dict_evaluation['sarcasm'] = evaluateClassification_sarcasm(y_true_sarcasm, y_pred_literal, y_pred_deep)
    return loss, dict_evaluation

class mainModel(object):
    def __init__(self):
        logging.basicConfig(
            filename='log/%s%s%s.log' % (FLAGS.name_dataset, FLAGS.name_model, time.strftime('%m-%d_%H.%M', time.localtime())),
            level=logging.DEBUG,
            format='%(asctime)s %(filename)s %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S'
        )

        self.dataset_name = ['train', 'test', 'valid'] # 'train', 'test', 'valid'
        self.datamanager = DataManager(FLAGS)
        self.data = {}
        self.data_predict = {}
        vocab, embed, vocab_dict = [], [], {}
        if int(FLAGS.predict) == 0:  # train mode
            for tmp in self.dataset_name:
                self.data[tmp] = self.datamanager.load_data(FLAGS.data_dir, f'{tmp}.txt')
                # self.data[tmp] = self.datamanager.load_data_sarcasm(FLAGS.data_dir, f'{tmp}.txt')
            vocab, embed, vocab_dict, pos_vocab = self.datamanager.build_vocab('%s%s' % (FLAGS.data_dir, FLAGS.path_wordvec), self.data[self.dataset_name[0]] + self.data[self.dataset_name[1]] + self.data[self.dataset_name[2]] )
        else:  # predict mode
            # predict with file
            for tmp in self.dataset_name:
                self.data[tmp] = self.datamanager.load_data(FLAGS.data_dir, f'{tmp}.txt')
            # vocab, embed = self.datamanager.load_vocab_embed('./twitter/', './twitter/')  # make sure the shape of embedding is the same with the checkpoint model
            vocab, embed, vocab_dict, pos_vocab = self.datamanager.build_vocab(
                '%s%s' % (FLAGS.data_dir, FLAGS.path_wordvec),
                self.data[self.dataset_name[0]] + self.data[self.dataset_name[1]] + self.data[self.dataset_name[2]])

        logging.info('model parameters: %s' % str(FLAGS))
        logging.info(f'Use device: {str(FLAGS.device)}')

        self.model = bridgeModel(FLAGS,
                vocab=vocab,
                embed=embed,
                #pos_vocab=pos_vocab
                )

    def train(self, breakpoint=-1):
        configure('summary/%s%s%s' % (FLAGS.name_dataset, FLAGS.name_model, time.strftime('%m-%d_%H.%M', time.localtime())), flush_secs=3)
        if breakpoint > 0:
            self.model.load_model('./models/%s%s' % (FLAGS.name_dataset, FLAGS.name_model), FLAGS.breakpoint)

        start_iter = 0 if breakpoint < 0 else (breakpoint * FLAGS.per_checkpoint + 1)
        loss_step, time_step = np.ones((num_loss, )), 0
        start_time = time.time()

        for step in range(start_iter, FLAGS.iter_num):
            if step % FLAGS.per_checkpoint == 0:
                show = lambda a: '[%s]' % (' '.join(['%.4f' % x for x in a]))
                time_step = time.time() - start_time
                logging.info('-'*50)
                logging.info('Time of iter training %.2f s' % time_step)
                logging.info('On iter step %s:, global step %d Loss-step %s' % (step/FLAGS.per_checkpoint, step, show(np.exp(loss_step)))) # step=k*per_checkpoint
                if FLAGS.save_model:
                    self.model.save_model('%s%s%s' % (FLAGS.model_dir, FLAGS.name_dataset, FLAGS.name_model), int(step/FLAGS.per_checkpoint))

                for name in self.dataset_name: # train valid test
                    loss, dict_eva = evaluate(self.model, self.datamanager, self.data[name])
                    log_value(f'Loss-{name}-all', loss[0], int(step/FLAGS.per_checkpoint))
                    dict_name = {'senti': 'senti', 'sarcasm': 'sarcasm', 'twitter': 'twitter', 'pair': 'pair', 'literal': 'literal', 'deep': 'deep'}
                    dict_keys = ['sarcasm', 'literal', 'deep']
                    logging.info(f'In dataset {name}: Loss is {show(loss)}') # ???
                    for i, tmp in enumerate(dict_keys):
                        log_value(f'Loss-{dict_name[tmp]}-{name}', loss[i+1], int(step / FLAGS.per_checkpoint))
                        log_value(f'Acc-{dict_name[tmp]}-{name}', dict_eva[tmp]['acc'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'F1-micro-{dict_name[tmp]}-{name}', dict_eva[tmp]['f1_micro'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'F1-macro-{dict_name[tmp]}-{name}', dict_eva[tmp]['f1_macro'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'pre-{dict_name[tmp]}-{name}', dict_eva[tmp]['pre_macro'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'rec-{dict_name[tmp]}-{name}', dict_eva[tmp]['rec_macro'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'AUC-{dict_name[tmp]}-{name}', dict_eva[tmp]['auc'],
                                  int(step / FLAGS.per_checkpoint))
                        logging.info(
                            f"\t\tAcc is {dict_eva[tmp]['acc']:.4f}, F1-micro is {dict_eva[tmp]['f1_micro']:.4f}")
                        logging.info(
                            f"\t\tF1-macro is {dict_eva[tmp]['f1_macro']:.4f}, AUC is {dict_eva[tmp]['auc']:.4f}")
                        logging.info(
                            f"\t\tPre-macro is {dict_eva[tmp]['pre_macro']:.4f}, Rec_macro is {dict_eva[tmp]['rec_macro']:.4f}")
                        logging.info(f"\t\tFor {tmp}, C_M is \n{dict_eva[tmp]['c_m']}")

                start_time = time.time()
                loss_step = np.zeros((num_loss, ))

            loss_step += train(self.model, self.datamanager, self.data[self.dataset_name[0]]) / FLAGS.per_checkpoint

    def predict(self, path, idx):
        self.model.load_model(path, idx)
        predict(self.model, self.datamanager, self.data['test'])


if __name__ == '__main__':
    mm = mainModel()
    if int(FLAGS.predict == 0):
        mm.train(FLAGS.breakpoint)
    else:
        mm.predict('%s' % (FLAGS.model_dir), FLAGS.breakpoint)
