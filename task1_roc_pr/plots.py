import json
import matplotlib.pyplot as plt

with open('task1_roc_pr/cnn_only_roc_pr_curve_data.json') as f:
    cnn_only = json.load(f)

with open('task1_roc_pr/efficientnet_b3_roc_pr_curve_data.json') as f:
    pretrained_cnn_only = json.load(f)

with open('task1_roc_pr/uni_only_roc_pr_curve_data.json') as f:
    uni_only = json.load(f)

with open('task1_roc_pr/hybrid_pretrained_roc_pr_curve_data.json') as f:
    hybrid_pretrained = json.load(f)

with open('task1_roc_pr/hybrid_scratch_roc_pr.json') as f:
    hybrid_scratch = json.load(f)

# ROC Curve
plt.plot(cnn_only["roc"]["fpr"], cnn_only["roc"]["tpr"], label="CNN (scratch)", linewidth=1)
plt.plot(pretrained_cnn_only["roc"]["fpr"], pretrained_cnn_only["roc"]["tpr"], label="CNN (pretrained)", linewidth=1)
plt.plot(uni_only["roc"]["fpr"], uni_only["roc"]["tpr"], label="UNI", linewidth=1)
plt.plot(hybrid_pretrained["roc"]["fpr"], hybrid_pretrained["roc"]["tpr"], label="Hybrid (pretrained)", linewidth=1)
plt.plot(hybrid_scratch["roc"]["fpr"], hybrid_scratch["roc"]["tpr"], label="Hybrid (scratch)", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Task 1 ROC Curves")
plt.legend()
plt.grid(True, alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# Precision-Recall Curve
plt.plot(cnn_only["pr"]["recall"], cnn_only["pr"]["precision"], label="CNN (scratch)", linewidth=1)
plt.plot(pretrained_cnn_only["pr"]["recall"], pretrained_cnn_only["pr"]["precision"], label="CNN (pretrained)", linewidth=1)
plt.plot(uni_only["pr"]["recall"], uni_only["pr"]["precision"], label="UNI", linewidth=1)
plt.plot(hybrid_pretrained["pr"]["recall"], hybrid_pretrained["pr"]["precision"], label="Hybrid (pretrained)", linewidth=1)
plt.plot(hybrid_scratch["pr"]["recall"], hybrid_scratch["pr"]["precision"], label="Hybrid (scratch)", linewidth=1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Task 1 Precision-Recall Curves")
plt.legend()
plt.grid(True, alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()