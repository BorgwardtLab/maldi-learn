"""
Implement own metric functions.

very_major_error_score:
    Calculates very major error.

major_error_score:
    Calculates major error.
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics._ranking import _binary_clf_curve




def specificity_score(y_true, y_pred, labels=[0,1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return TN / float(TN + FP)


def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, average='binary') 






def very_major_error_score(y_true, y_pred, labels=[0,1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return FP / float(FP+TN)

def major_error_score(y_true, y_pred, labels=[0,1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return FN / float(FN+TP)


def vme_curve(y_true, y_pred, pos_label=1, sample_weight=None):

    fps, tps, thresholds = _binary_clf_curve(y_true,
                                             y_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)

    vme = fps / fps[-1]
    me = (tps[-1] - tps) / tps[-1]
    return vme, 1-me, thresholds


def vme_auc_score(y_true, y_pred, pos_label=1, sample_weight=None):
    vme, me_inv, thresholds = vme_curve(y_true,
                                        y_pred,
                                        pos_label=1,
                                        sample_weight=None)
    return auc(vme, me_inv)
