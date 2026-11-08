"""
Evaluation metrics computation module.
Computes Accuracy, Precision, Recall, F1-Score (macro & weighted),
Confusion Matrix, and inference latency.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import config

def compute_classification_metrics(y_true, y_pred, model_name="Model", train_time=0.0, inference_time_ms=0.0):
    """
    Computes standard multi-class metrics for the 20 LEGO categories.
    """
    acc = float(accuracy_score(y_true, y_pred))
    
    # Macro metrics (unweighted average across all classes)
    p_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    # Weighted metrics (weighted by support)
    p_weighted = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    
    report_dict = classification_report(
        y_true, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.ALL_CLASSES,
        output_dict=True,
        zero_division=0
    )
    
    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "train_time_sec": float(train_time),
        "inference_ms_sample": float(inference_time_ms),
        "confusion_matrix": cm.tolist(),
        "per_class_report": report_dict
    }

