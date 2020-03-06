"""
Implement own metric functions.
"""

from sklearn.metrics import confusion_matrix

def very_major_error_score(y_true, y_pred, labels=[0,1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return FP / float(FP+TN)

