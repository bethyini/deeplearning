import json
import matplotlib.pyplot as plt

with open('task1_roc_pr/normalization_experiments/hybrid_baseline_roc_pr.json') as f:
    hybrid_baseline = json.load(f)

with open('task1_roc_pr/normalization_experiments/hybrid_norm_roc_pr.json') as f:
    hybrid_norm = json.load(f)

# ROC Curve
plt.plot(hybrid_baseline["roc"]["fpr"], hybrid_baseline["roc"]["tpr"], label="Baseline", linewidth=1)
plt.plot(hybrid_norm["roc"]["fpr"], hybrid_norm["roc"]["tpr"], label="Normalized", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Normalization Experiment ROC Curves")
plt.legend()
plt.grid(True, alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# Precision-Recall Curve
plt.plot(hybrid_baseline["pr"]["recall"], hybrid_baseline["pr"]["precision"], label="Baseline", linewidth=1)
plt.plot(hybrid_norm["pr"]["recall"], hybrid_norm["pr"]["precision"], label="Normalized", linewidth=1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Normalization Experiment Precision-Recall Curves")
plt.legend()
plt.grid(True, alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()