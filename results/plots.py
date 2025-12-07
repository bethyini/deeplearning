import matplotlib.pyplot as plt
import json

with open("hybrid/exp1_baseline_training_history.json") as f:
  baseline = json.load(f)

# ROC Curve
plt.plot(baseline["fpr"], baseline["tpr"], label="ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True, alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# Precision-Recall curve
plt.figure(figsize=(7,6))
plt.plot(baseline["val_recall"], baseline["val_precision"], linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()