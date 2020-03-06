"""
Implement own metric functions.

very_major_error_score:
    Calculates very major error, also called false positive rate.

major_error_score:
    Calculates major error, also called false negative rate.
"""

from sklearn.metrics import confusion_matrix

def very_major_error_score(y_true, y_pred, labels=[0,1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return FP / float(FP+TN)

def major_error_score(y_true, y_pred, labels=[0,1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return FN / float(FN+TP)
