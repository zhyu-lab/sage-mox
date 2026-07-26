# -*- coding:utf-8 -*-

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import (
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
)
from scipy.optimize import linear_sum_assignment


def align_pred_to_true(y_true, y_pred):
    """
    Match predicted cluster ids to ground-truth label ids (Hungarian on overlap counts).
    Returns y_pred remapped so each cluster uses the matched true label id.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    true_ids = np.unique(y_true)
    pred_ids = np.unique(y_pred)
    n_true, n_pred = len(true_ids), len(pred_ids)
    if n_true == 0 or n_pred == 0:
        return y_pred.copy()

    cost = np.zeros((n_true, n_pred), dtype=np.int64)
    for i, t in enumerate(true_ids):
        mask_t = y_true == t
        for j, p in enumerate(pred_ids):
            cost[i, j] = int(np.sum(mask_t & (y_pred == p)))

    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {pred_ids[j]: true_ids[i] for i, j in zip(row_ind, col_ind)}

    for j, p in enumerate(pred_ids):
        if p not in mapping:
            i = int(np.argmax(cost[:, j]))
            mapping[p] = true_ids[i]

    aligned = np.empty_like(y_pred)
    for p, t in mapping.items():
        aligned[y_pred == p] = t
    return aligned


def eval(label, pred):
    ari = ari_score(label, pred)
    nmi = nmi_score(label, pred, average_method="arithmetic")
    ami = adjusted_mutual_info_score(label, pred)
    v_measure = v_measure_score(label, pred)
    homo = homogeneity_score(label, pred)
    return ari, nmi, ami, v_measure, homo
