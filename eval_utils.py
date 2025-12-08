import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

def collect_probs_labels(model, val_loader, device, batch_to_logits, positive_class=1):
    """
    Generic helper:
      - model: nn.Module
      - val_loader: DataLoader
      - device: torch.device
      - batch_to_logits: function(model, batch, device) -> (logits, labels)
      - positive_class: index of positive class in logits

    Returns:
      all_probs: np.array of shape (N,)
      all_labels: np.array of shape (N,)
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            logits, labels = batch_to_logits(model, batch, device)
            # logits: (B, C), labels: (B,)
            probs = F.softmax(logits, dim=1)
            all_probs.extend(probs[:, positive_class].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def compute_roc_pr(all_labels, all_probs):
    """
    all_labels, all_probs: 1D numpy arrays
    Returns dict with ROC + PR curve data and summary metrics.
    """
    # ROC
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)

    # Precision–Recall
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)

    return {
        "roc": {
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "auc": roc_auc,
        },
        "pr": {
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
            "average_precision": ap,
        },
    }


def get_curve_data(model, val_loader, device, batch_to_logits, positive_class=1):
    """
    Model-agnostic ROC + PR curve computation.
    """
    all_probs, all_labels = collect_probs_labels(
        model, val_loader, device, batch_to_logits, positive_class=positive_class
    )
    curves = compute_roc_pr(all_labels, all_probs)
    return curves


def save_curve_data(curves, filepath):
    """
    Save ROC + PR curve data to JSON file.
    """
    import json

    curve_data = {
        "roc": {
            "fpr": curves["roc"]["fpr"].tolist(),
            "tpr": curves["roc"]["tpr"].tolist(),
            "roc_thresholds": curves["roc"]["roc_thresholds"].tolist(),
            "auc": float(curves["roc"]["auc"]),
        },
        "pr": {
            "precision": curves["pr"]["precision"].tolist(),
            "recall": curves["pr"]["recall"].tolist(),
            "pr_thresholds": curves["pr"]["pr_thresholds"].tolist(),
            "average_precision": float(curves["pr"]["average_precision"]),
        },
    }

    with open(filepath, "w") as f:
        json.dump(curve_data, f, indent=2)
    print(f"Saved PR+ROC data to {filepath}")

