"""XHAR metric definitions, extracted from XHAR-SDCN b4f27c6.

cluster_acc is copied verbatim; logging and wandb are omitted.
Labels enter this evaluation module only, never model or objective construction.
"""
import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def cluster_acc(y_true, y_pred):
    # Normalize true labels to start from 0
    y_true = y_true - np.min(y_true)

    # Get unique class/cluster IDs
    l1 = list(np.unique(y_true))   # ground-truth labels
    l2 = list(np.unique(y_pred))   # predicted cluster labels
    numclass1, numclass2 = len(l1), len(l2)

    # Build cost matrix: cost[i][j] = number of samples where y_true==l1[i] and y_pred==l2[j]
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mask_true = (y_true == c1)
        for j, c2 in enumerate(l2):
            cost[i, j] = np.sum(mask_true & (y_pred == c2))

    # Pad cost matrix to be square if needed (Hungarian requires rectangular OK, but safer)
    if numclass1 > numclass2:
        cost = np.hstack([cost, np.zeros((numclass1, numclass1 - numclass2), dtype=int)])
        l2 += [None] * (numclass1 - numclass2)
    elif numclass2 > numclass1:
        cost = np.vstack([cost, np.zeros((numclass2 - numclass1, numclass2), dtype=int)])
        l1 += [None] * (numclass2 - numclass1)

    # Hungarian algorithm expects a cost matrix to minimize — we use negative counts
    m = Munkres()
    indexes = m.compute(cost.__neg__().tolist())

    # Build remapping dict: predicted_label -> matched true_label
    mapping = {}
    for i, j in indexes:
        if l1[i] is not None and l2[j] is not None:
            mapping[l2[j]] = l1[i]

    # Apply the best matching
    new_predict = np.zeros_like(y_pred)
    for pred_label in l2:
        if pred_label is None:  # skip padded labels
            continue
        mask = (y_pred == pred_label)
        if pred_label in mapping:
            new_predict[mask] = mapping[pred_label]
        else:
            # Unmatched cluster, assign a dummy label
            new_predict[mask] = -1

    # Compute metrics
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro', zero_division=0)
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro', zero_division=0)
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro', zero_division=0)
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro', zero_division=0)
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro', zero_division=0)
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro', zero_division=0)

    return acc, f1_macro, f1_micro, precision_macro, precision_micro, recall_macro, recall_micro


def evaluate_clusters(labels, predicted):
    values = cluster_acc(np.asarray(labels), np.asarray(predicted))
    names = ["Accuracy", "Macro F1", "Micro F1", "Macro Precision", "Micro Precision", "Macro Recall", "Micro Recall"]
    result = dict(zip(names, map(float, values)))
    result["NMI"] = float(normalized_mutual_info_score(labels, predicted, average_method="arithmetic"))
    result["ARI"] = float(adjusted_rand_score(labels, predicted))
    return result
