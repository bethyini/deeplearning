"""
Shared training utilities
"""
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, roc_curve, precision_recall_curve)
from pathlib import Path


def compute_metrics(total_loss, all_labels, all_preds, all_probs, loader):
    "compute metrics"
    # compute roc curve
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)

    # compute precision recall curve
    precision, recall, prc_thresholds = precision_recall_curve(all_labels, all_probs)

    # compute metrics
    metrics = {
        'loss': total_loss/len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_thresholds': roc_thresholds.tolist(),
        'precision_curve': precision.tolist(),
        'recall_curve': recall.tolist(),
        'prc_thresholds': prc_thresholds.tolist()
    }
    return metrics

def initialize_history():
    """
    Initialize empty history dict to track metrics
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_balanced_acc": [],
        "val_balanced_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_roc_auc": [],
        "val_fpr": [],
        "val_tpr": [],
        "val_roc_thresholds": [],
        "val_precision_curve": [],
        "val_recall_curve": [],
        "val_prc_thresholds": [],
        'lr': [],
        'epoch_time': []
    }
    return history

def update_history(history, train_loss, train_acc, train_bal_acc, val_metrics, lr, epoch_time):
    """
    Update history with metrics from current epoch
    
    Args:
        history: History dictionary
        train_loss: Training loss
        train_acc: Training accuracy
        train_bal_acc: Training balanced accuracy
        val_metrics: Dictionary of validation metrics from validate_*()
        lr: Current learning rate
        epoch_time: Time taken for epoch
    """
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_balanced_acc'].append(train_bal_acc)
    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_roc_auc'].append(val_metrics['roc_auc'])
    history['val_fpr'].append(val_metrics['fpr'])
    history['val_tpr'].append(val_metrics['tpr'])
    history['val_roc_thresholds'].append(val_metrics['roc_thresholds'])
    history['val_precision_curve'].append(val_metrics['precision_curve'])
    history['val_recall_curve'].append(val_metrics['recall_curve'])
    history['val_prc_thresholds'].append(val_metrics['prc_thresholds'])
    history['lr'].append(lr)
    history['epoch_time'].append(epoch_time)


def save_history(history, filepath, epoch=None):
    """
    Save training history to json after each epoch
    """
    filepath = Path(filepath)
    # write to temp file to avoid corruption
    temp_file = filepath.parent / f"{filepath.stem}_temp.json"
    with open(temp_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # atomic rename to replace old file
    temp_file.replace(filepath)


def save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, filepath, 
                   save_optimizer=True):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        val_metrics: Validation metrics dictionary
        config: Configuration dictionary
        filepath: Path to save checkpoint
        save_optimizer: Whether to save optimizer/scheduler state
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_balanced_acc': val_metrics['balanced_accuracy'],
        'val_roc_auc': val_metrics['roc_auc'],
        'val_f1': val_metrics['f1'],
        'config': config,
    }
    
    if save_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)


def train_epoch_uni(model, loader, criterion, optimizer, device):
    """
    train any model with single input (not just uni) for one epoch
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc='training'):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()  # zero the optimizer
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return total_loss/len(loader), acc, bal_acc

def train_epoch_hybrid(model, loader, criterion, optimizer, device):
    """
    train hybrid model for one epoch
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for img_uni, img_cnn, labels in tqdm(loader, desc='train'):
        img_uni, img_cnn, labels = img_uni.to(device), img_cnn.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(img_uni, img_cnn)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return total_loss/len(loader), acc, bal_acc

def validate_uni(model, loader, criterion, device):
    """
    validate uni (but really any) model
    """
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='val'):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return compute_metrics(total_loss, all_labels, all_preds, all_probs, loader)


def validate_hybrid(model, loader, criterion, device):
    "validate hyrbid model"
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for img_uni, img_cnn, labels in tqdm(loader, desc='val'):
            img_uni, img_cnn, labels = img_uni.to(device), img_cnn.to(device), labels.to(device)
            logits, _ = model(img_uni, img_cnn)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.nn.functional.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return compute_metrics(total_loss, all_labels, all_preds, all_probs, loader)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def count_parameters(model):
    """
    count trainable and total parameters
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

