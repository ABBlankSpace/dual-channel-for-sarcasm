# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

def evaluateClassification(list_t, list_prob_pred):
    list_p = np.argmax(list_prob_pred, axis=-1)
    assert len(list_t) == len(list_p)
    n_class = np.size(list_prob_pred, 1)
    c_m = confusion_matrix(list_t, list_p)
    acc = accuracy_score(list_t, list_p)
    # pre_micro = precision_score(list_t, list_p, average='micro')
    # pre_weighted = precision_score(list_t, list_p, average='weighted')
    pre_macro = precision_score(list_t, list_p, average='macro')
    # rec_micro = recall_score(list_t, list_p, average='micro')
    # rec_weighted = recall_score(list_t, list_p, average='weighted')
    rec_macro = recall_score(list_t, list_p, average='macro')
    f1_micro = f1_score(list_t, list_p, average='micro')
    f1_macro = f1_score(list_t, list_p, average='macro')
    auc_list = []
    for i in range(n_class):
        fpr, tpr, thresholds = roc_curve(list_t, list_p, pos_label=i)
        auc_list.append(auc(fpr, tpr))
    # f1_weighted = f1_score(list_t, list_p, average='weighted')
    # f1 = f1_score(list_t, list_p, average='samples')

    return {'c_m': c_m, 'acc': acc, 'f1_macro': f1_macro, 'pre_macro': pre_macro, 'rec_macro': rec_macro, 'f1_micro': f1_micro,
            'auc': np.mean(auc_list)}

def evaluateClassification_sarcasm(list_sarcasm, list_prob_pred_literal, list_prob_pred_deep):
    list_p_literal = np.argmax(list_prob_pred_literal, axis=-1)
    list_p_deep = np.argmax(list_prob_pred_deep, axis=-1)
    list_p_sarcasm = []
    for i in range(len(list_p_literal)):
        if int(list_p_literal[i]+list_p_deep[i]) == 1: # 0 1 or 1 0
            list_p_sarcasm.append(1)
        else:
            list_p_sarcasm.append(0)
    assert len(list_sarcasm) == len(list_p_literal) == len(list_p_deep)
    n_class = np.size(list_prob_pred_deep, 1)
    c_m = confusion_matrix(list_sarcasm, list_p_sarcasm)
    acc = accuracy_score(list_sarcasm, list_p_sarcasm)
    # pre_micro = precision_score(list_t, list_p, average='micro')
    # pre_weighted = precision_score(list_t, list_p, average='weighted')
    pre_macro = precision_score(list_sarcasm, list_p_sarcasm, average='macro')
    # rec_micro = recall_score(list_t, list_p, average='micro')
    # rec_weighted = recall_score(list_t, list_p, average='weighted')
    rec_macro = recall_score(list_sarcasm, list_p_sarcasm, average='macro')
    f1_micro = f1_score(list_sarcasm, list_p_sarcasm, average='micro')
    f1_macro = f1_score(list_sarcasm, list_p_sarcasm, average='macro')
    auc_list = []
    for i in range(n_class):
        fpr, tpr, thresholds = roc_curve(list_sarcasm, list_p_sarcasm, pos_label=i)
        auc_list.append(auc(fpr, tpr))
    # f1_weighted = f1_score(list_t, list_p, average='weighted')
    # f1 = f1_score(list_t, list_p, average='samples')

    return {'c_m': c_m, 'acc': acc, 'f1_macro': f1_macro, 'pre_macro': pre_macro, 'rec_macro': rec_macro, 'f1_micro': f1_micro,
            'auc': np.mean(auc_list)}

if __name__ == '__main__':
    np.random.seed(2019)
    y_true_M = np.random.randint(0, 3, 10)
    # y_pred_M = np.array([1, 0, 2, 0, 1, 1, 0, 1, 0, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0], dtype='float64')
    y_pred_prob = []
    for i in range(len(y_true_M)):
        y_pred_prob.append(np.random.random(3))
    print(y_true_M, y_pred_prob)
    print(evaluateClassification(y_true_M, y_pred_prob))
    # print(evaluateUnbalancedClassification(y_true_M, y_pred_prob, 1))