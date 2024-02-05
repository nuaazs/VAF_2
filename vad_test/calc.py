import numpy as np

def calc_confusion_matrix(y_true, y_pred):
    # y_true: [1,1,1,0,0,0...]
    # y_pred: [1,0,1,0,0,0...]
    # Calaculate confusion matrix
    # TN, FP, FN, TP
    # fast calc
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tn, fp, fn, tp

def calc_all(y_true, y_pred):
    # y_true: [1,1,1,0,0,0...]
    # y_pred: [1,0,1,0,0,0...]
    # Calaculate all metrics
    # TN, FP, FN, TP, TPR, FPR, ACC, PRE, REC, F1
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn, fp, fn, tp = calc_confusion_matrix(y_true, y_pred)
    tpr = tp / (tp + fn + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * pre * rec / (pre + rec + 1e-6)
    result_dict = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'tpr': tpr,
        'fpr': fpr,
        'acc': acc,
        'pre': pre,
        'rec': rec,
        'f1': f1
    }