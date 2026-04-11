import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve,
    precision_score, recall_score, f1_score,
)


def compute_metrics(y_true, y_scores, threshold=0.5):
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: np.array of 0/1 labels (0=real, 1=ai)
        y_scores: np.array of probabilities (AI class)
        threshold: classification threshold

    Returns:
        dict with all metrics
    """
    y_pred = (y_scores > threshold).astype(int)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Real', 'AI'])

    # ROC / AUC
    auc_roc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # EER: point where FPR == FNR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc_roc': auc_roc,
        'average_precision': ap,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm,
        'classification_report': report,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': thresholds,
    }


def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[eer_idx], thresholds[eer_idx]
