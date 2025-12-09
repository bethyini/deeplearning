import json

def print_metrics_for_best_epoch(history):
    best_idx = max(range(len(history['val_acc'])), key=lambda i: history['val_acc'][i])

    vals = [
        history['val_acc'][best_idx],
        history['val_precision'][best_idx],
        history['val_recall'][best_idx],
        history['val_roc_auc'][best_idx],
        history['val_f1'][best_idx],
    ]

    print(*(round(v, 3) for v in vals))

# Task 1
print('Task 1')
# UNI-only
with open('uni_only/uni2h_3mlp_training_history.json') as f:
    uni_only = json.load(f)
print_metrics_for_best_epoch(uni_only)

# CNN scratch
with open('cnn_only/scratch_cnn_only_training_history.json') as f:
    scratch_cnn = json.load(f)
print_metrics_for_best_epoch(scratch_cnn)

# CNN pretrained
with open('cnn_only/efficientnet_b3_training_history.json') as f:
    pretrained_cnn = json.load(f)
print_metrics_for_best_epoch(pretrained_cnn)

# Hybrid pretrained
with open('hybrid/hybrid_pretrained_training_history.json') as f:
    hybrid_pretrained = json.load(f)
print_metrics_for_best_epoch(hybrid_pretrained)

# Hybrid scratch
with open('hybrid/hybrid_scratch_training_history.json') as f:
    hybrid_scratch = json.load(f)
print_metrics_for_best_epoch(hybrid_scratch)


def print_metrics_for_best_epoch2(history):
    best_idx = max(range(len(history['val_balanced_acc'])), key=lambda i: history['val_balanced_acc'][i])

    vals = [
        history['val_balanced_acc'][best_idx],
        history['val_precision'][best_idx],
        history['val_recall'][best_idx],
        history['val_roc_auc'][best_idx],
        history['val_f1'][best_idx],
    ]

    print(*(round(v, 3) for v in vals))


# Task 2
print('Task 2')
# UNI-only
with open('task2/amibr_uni_history.json') as f:
    uni_only2 = json.load(f)
print_metrics_for_best_epoch2(uni_only2)

# CNN scratch
with open('task2/amibr_cnn_scratch_history.json') as f:
    scratch_cnn2 = json.load(f)
print_metrics_for_best_epoch2(scratch_cnn2)

# CNN pretrained
with open('task2/amibr_efficientnet_history.json') as f:
    pretrained_cnn2 = json.load(f)
print_metrics_for_best_epoch2(pretrained_cnn2)

# Hybrid scratch
with open('task2/amibr_hybrid_scratch.json') as f:
    hybrid_scratch2 = json.load(f)
print_metrics_for_best_epoch2(hybrid_scratch2)


# Normalization Experiment
print('Normalization Experiment')

# Baseline
with open('hybrid/exp1_baseline_training_history.json') as f:
    baseline = json.load(f)
print_metrics_for_best_epoch(baseline)

# With Normalization
with open('hybrid/exp1_norm_hybrid_training_history.json') as f:
    normalized = json.load(f)
print_metrics_for_best_epoch(normalized)