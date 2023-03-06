import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc


def get_y(final_list):
    y_gt, y_pred = [], []
    for item in final_list:
        gt_list = item[0]
        pred_list = item[1]
        for i in range(len(gt_list)):
            y_gt.append(gt_list[i])
            y_pred.append((pred_list[i]))
    return y_gt, y_pred


def get_fpr_at_95_tpr(tpr, fpr):
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return 1 - min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return 1 - np.interp(0.95, tpr, fpr)


def cal_auroc(final_list):
    y_gt, y_pred = get_y(final_list)
    auc = metrics.roc_auc_score(y_gt, y_pred)
    if auc > 0.5:
        return auc
    else:
        return 1 - auc


def cal_aupr(final_list):
    y_gt, y_pred = get_y(final_list)
    return average_precision_score(y_gt, y_pred)


def cal_acc(final_list):
    y_gt, y_pred = get_y(final_list)
    return accuracy_score(y_gt, np.round(y_pred))


def cal_det_err(final_list, alpha=0.5):
    y_gt, y_pred = get_y(final_list)
    fpr, tpr, thresholds = roc_curve(y_gt, y_pred)
    return min(alpha * (1 - tpr) + (1 - alpha) * fpr)


def cal_f1(final_list):
    y_gt, y_pred = get_y(final_list)
    return f1_score(y_gt, np.round(y_pred))


def cal_fpr_at_95_tpr(final_list):
    y_gt, y_pred = get_y(final_list)
    fpr, tpr, thresholds = roc_curve(y_gt, y_pred)
    return get_fpr_at_95_tpr(tpr, fpr)


