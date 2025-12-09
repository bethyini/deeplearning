import json
import matplotlib.pyplot as plt
import copy

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

# write json files with best epochs only
curve_keys = [
    "val_fpr",
    "val_tpr",
    "val_roc_thresholds",
    "val_precision_curve",
    "val_recall_curve",
    "val_prc_thresholds",
]

cnn_flat = copy.deepcopy(cnn_only)
for key in curve_keys:
    cnn_flat[key] = cnn_only[key][best_epoch_cnn_only]
with open('task2/amibr_cnn_scratch_best_epoch.json', 'w') as f:
    json.dump(cnn_flat, f, indent=2)

pretrained_cnn_flat = copy.deepcopy(pretrained_cnn_only)
for key in curve_keys:
    pretrained_cnn_flat[key] = pretrained_cnn_only[key][best_epoch_pretrained_cnn]
with open('task2/amibr_efficientnet_best_epoch.json', 'w') as f:
    json.dump(pretrained_cnn_flat, f, indent=2)

uni_flat = copy.deepcopy(uni_only)
for key in curve_keys:
    uni_flat[key] = uni_only[key][best_epoch_uni_only]
with open('task2/amibr_uni_best_epoch.json', 'w') as f:
    json.dump(uni_flat, f, indent=2)

hybrid_scratch_flat = copy.deepcopy(hybrid_scratch)
for key in curve_keys:
    hybrid_scratch_flat[key] = hybrid_scratch[key][best_epoch_hybrid_scratch]
with open('task2/amibr_hybrid_scratch_best_epoch.json', 'w') as f:
    json.dump(hybrid_scratch_flat, f, indent=2)

plt.figure()
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

plt.figure()
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