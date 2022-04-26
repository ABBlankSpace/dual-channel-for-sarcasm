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
    pre_macro = precision_score(list_t, list_p, average='macro')
    rec_macro = recall_score(list_t, list_p, average='macro')
    f1_micro = f1_score(list_t, list_p, average='micro')
    f1_macro = f1_score(list_t, list_p, average='macro')
    auc_list = []
    for i in range(n_class):
        fpr, tpr, thresholds = roc_curve(list_t, list_p, pos_label=i)
        auc_list.append(auc(fpr, tpr))

    return {'c_m': c_m, 'acc': acc, 'f1_macro': f1_macro, 'pre_macro': pre_macro, 'rec_macro': rec_macro, 'f1_micro': f1_micro,
            'auc': np.mean(auc_list)}