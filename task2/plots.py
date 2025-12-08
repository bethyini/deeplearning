import json
import matplotlib.pyplot as plt

with open('task2/amibr_cnn_scratch_history.json') as f:
    cnn_only = json.load(f)

with open('task2/amibr_efficientnet_history.json') as f:
    pretrained_cnn_only = json.load(f)

with open('task2/amibr_uni_history.json') as f:
    uni_only = json.load(f)

with open('task2/amibr_hybrid_scratch.json') as f:
    hybrid_scratch = json.load(f)

# best epochs (based on val balanced accuracy)
val_bal_acc_cnn_only = cnn_only["val_balanced_acc"]
best_epoch_cnn_only = max(range(len(val_bal_acc_cnn_only)), key=lambda i: val_bal_acc_cnn_only[i])

val_bal_acc_pretrained_cnn = pretrained_cnn_only["val_balanced_acc"]
best_epoch_pretrained_cnn = max(range(len(val_bal_acc_pretrained_cnn)), key=lambda i: val_bal_acc_pretrained_cnn[i])

val_bal_acc_uni_only = uni_only["val_balanced_acc"]
best_epoch_uni_only = max(range(len(val_bal_acc_uni_only)), key=lambda i: val_bal_acc_uni_only[i])

val_bal_acc_hybrid_scratch = hybrid_scratch["val_balanced_acc"]
best_epoch_hybrid_scratch = max(range(len(val_bal_acc_hybrid_scratch)), key=lambda i: val_bal_acc_hybrid_scratch[i])

# ROC Curves
plt.plot(cnn_only["val_fpr"][best_epoch_cnn_only], cnn_only["val_tpr"][best_epoch_cnn_only], label="CNN (scratch)", linewidth=1)
plt.plot(pretrained_cnn_only["val_fpr"][best_epoch_pretrained_cnn], pretrained_cnn_only["val_tpr"][best_epoch_pretrained_cnn], label="CNN (pretrained)", linewidth=1)
plt.plot(uni_only["val_fpr"][best_epoch_uni_only], uni_only["val_tpr"][best_epoch_uni_only], label="UNI", linewidth=1)
plt.plot(hybrid_scratch["val_fpr"][best_epoch_hybrid_scratch], hybrid_scratch["val_tpr"][best_epoch_hybrid_scratch], label="Hybrid (scratch)", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Task 2 ROC Curves")
plt.legend()
plt.grid(True, alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# Precision-Recall Curves
plt.plot(cnn_only["val_recall_curve"][best_epoch_cnn_only], cnn_only["val_precision_curve"][best_epoch_cnn_only], label="CNN (scratch)", linewidth=1)
plt.plot(pretrained_cnn_only["val_recall_curve"][best_epoch_pretrained_cnn], pretrained_cnn_only["val_precision_curve"][best_epoch_pretrained_cnn], label="CNN (pretrained)", linewidth=1)
plt.plot(uni_only["val_recall_curve"][best_epoch_uni_only], uni_only["val_precision_curve"][best_epoch_uni_only], label="UNI", linewidth=1)
plt.plot(hybrid_scratch["val_recall_curve"][best_epoch_hybrid_scratch], hybrid_scratch["val_precision_curve"][best_epoch_hybrid_scratch], label="Hybrid (scratch)", linewidth=1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Task 2 Precision-Recall Curves")
plt.legend()
plt.grid(True, alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()