import json
import numpy as np
import matplotlib.pyplot as plt

with open("uni_only/uni2h_3mlp_training_history.json") as f:
  uni_only = json.load(f)

with open("cnn_only/efficientnet_b3_training_history.json") as f:
  cnn_only = json.load(f)

with open("hybrid/hybrid_pretrained_training_history.json") as f:
  hybrid_pretrained = json.load(f)

with open("hybrid/hybrid_scratch_training_history.json") as f:
  hybrid_scratch = json.load(f)

epochs = list(range(1, len(uni_only['train_loss']) + 1))

'''
# Training Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, uni_only['train_loss'], 'b-o', markersize=4, label='UNI-only')
plt.plot(epochs, cnn_only['train_loss'], 'g-o', markersize=4, label='CNN-only')
plt.plot(epochs, hybrid_pretrained['train_loss'], 'r-o', markersize=4, label='Hybrid (pretrained)')
plt.plot(epochs, hybrid_scratch['train_loss'], 'y-o', markersize=4, label='Hybrid (scratch)')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss')
plt.legend()
plt.tight_layout()
plt.show()


# Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, uni_only['val_loss'], 'b-o', markersize=4, label='UNI-only')
plt.plot(epochs, cnn_only['val_loss'], 'g-o', markersize=4, label='CNN-only')
plt.plot(epochs, hybrid_pretrained['val_loss'], 'r-o', markersize=4, label='Hybrid (pretrained)')
plt.plot(epochs, hybrid_scratch['val_loss'], 'y-o', markersize=4, label='Hybrid (scratch)')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Validation F1-Score
# Balance between precision and recall
plt.figure(figsize=(8, 5))
plt.plot(epochs, uni_only['val_f1'], 'b-o', markersize=4, label='UNI-only')
plt.plot(epochs, cnn_only['val_f1'], 'g-o', markersize=4, label='CNN-only')
plt.plot(epochs, hybrid_pretrained['val_f1'], 'r-o', markersize=4, label='Hybrid (pretrained)')
plt.plot(epochs, hybrid_scratch['val_f1'], 'y-o', markersize=4, label='Hybrid (scratch)')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score')
plt.legend()
plt.tight_layout()
plt.show()

# Validation ROC-AUC
# how well a model can distinguish positives from negatives across all possible classification thresholds
plt.figure(figsize=(8, 5))
plt.plot(epochs, uni_only['val_roc_auc'], 'b-o', markersize=4, label='UNI-only')
plt.plot(epochs, cnn_only['val_roc_auc'], 'g-o', markersize=4, label='CNN-only')
plt.plot(epochs, hybrid_pretrained['val_roc_auc'], 'r-o', markersize=4, label='Hybrid (pretrained)')
plt.plot(epochs, hybrid_scratch['val_roc_auc'], 'y-o', markersize=4, label='Hybrid (scratch)')
plt.xlabel('Epoch')
plt.ylabel('ROC-AUC')
plt.title('ROC-AUC')
plt.legend()
plt.tight_layout()
plt.show()
'''

# The plots on the same page
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

# Training Loss
axs[0].plot(epochs, uni_only['train_loss'], 'b-o', markersize=4, label='UNI-only')
axs[0].plot(epochs, cnn_only['train_loss'], 'g-o', markersize=4, label='CNN-only')
axs[0].plot(epochs, hybrid_pretrained['train_loss'], 'r-o', markersize=4, label='Hybrid (pretrained)')
axs[0].plot(epochs, hybrid_scratch['train_loss'], 'y-o', markersize=4, label='Hybrid (scratch)')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Training Loss')
axs[0].set_title('Training Loss')
axs[0].legend()

# Validation Loss
axs[1].plot(epochs, uni_only['val_loss'], 'b-o', markersize=4, label='UNI-only')
axs[1].plot(epochs, cnn_only['val_loss'], 'g-o', markersize=4, label='CNN-only')
axs[1].plot(epochs, hybrid_pretrained['val_loss'], 'r-o', markersize=4, label='Hybrid (pretrained)')
axs[1].plot(epochs, hybrid_scratch['val_loss'], 'y-o', markersize=4, label='Hybrid (scratch)')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Validation Loss')
axs[1].set_title('Validation Loss')
axs[1].legend()

# Validation F1 Score
axs[2].plot(epochs, uni_only['val_f1'], 'b-o', markersize=4, label='UNI-only')
axs[2].plot(epochs, cnn_only['val_f1'], 'g-o', markersize=4, label='CNN-only')
axs[2].plot(epochs, hybrid_pretrained['val_f1'], 'r-o', markersize=4, label='Hybrid (pretrained)')
axs[2].plot(epochs, hybrid_scratch['val_f1'], 'y-o', markersize=4, label='Hybrid (scratch)')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('F1 Score')
axs[2].set_title('F1 Score')
axs[2].legend()

# Validation ROC-AUC
axs[3].plot(epochs, uni_only['val_roc_auc'], 'b-o', markersize=4, label='UNI-only')
axs[3].plot(epochs, cnn_only['val_roc_auc'], 'g-o', markersize=4, label='CNN-only')
axs[3].plot(epochs, hybrid_pretrained['val_roc_auc'], 'r-o', markersize=4, label='Hybrid (pretrained)')
axs[3].plot(epochs, hybrid_scratch['val_roc_auc'], 'y-o', markersize=4, label='Hybrid (scratch)')
axs[3].set_xlabel('Epoch')
axs[3].set_ylabel('ROC-AUC')
axs[3].set_title('ROC-AUC')
axs[3].legend()

plt.tight_layout()
plt.show()