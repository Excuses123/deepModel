# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


from sklearn.metrics import roc_auc_score


def cal_group_auc(data, uid, labels, preds):
    """Calculate group auc"""
    total_auc = 0
    impression_total = 0
    for uid, df in data.groupby(uid):
        if 0 < df[labels].mean() < 1:
            auc = roc_auc_score(df[labels], df[preds])
            total_auc += auc * len(df)
            impression_total += len(df)
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)

    return group_auc


def mean_reciprocal_rank(label, pred):
    """
    MRR(https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
    """
    assert len(label) == len(pred)
    score_list = []
    for i, id in enumerate(label):
        p_list = pred[i]
        score = 0
        if id in p_list:
            score = 1 / (p_list.index(id) + 1)
        score_list.append(score)
    return sum(score_list) / len(label)
