import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score
)
from src.config import REPORTS_DIR, LABEL_COLUMN, TARGET_COLUMN, ATTACK_CATEGORIES

def optimize_threshold(model, X_val, y_val):
    """
    Scans classification thresholds from 0.01 to 0.99 exclusively on the validation set.
    Computes:
      1. F1-optimal threshold (standard balanced setting)
      2. Cost-sensitive balanced threshold (maximize Recall subject to validation FPR <= 0.50%)
      3. Cost-sensitive high-detection threshold (maximize Recall subject to validation FPR <= 1.00%)
    """
    print("\n--- Optimizing Classification Threshold on Validation Set ---")
    y_prob = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.linspace(0.01, 0.99, 99)
    records = []
    
    best_f1 = -1.0
    optimal_threshold_f1 = 0.50
    
    # Cost-sensitive: max recall with FPR <= 0.5%
    best_rec_cs_05 = -1.0
    thresh_cs_05 = 0.10
    
    # Cost-sensitive: max recall with FPR <= 1.0%
    best_rec_cs_10 = -1.0
    thresh_cs_10 = 0.05
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        records.append({
            'threshold': round(float(thresh), 2),
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'balanced_accuracy': round(float(bal_acc), 4),
            'fpr': round(float(fpr), 4),
            'fnr': round(float(fnr), 4)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold_f1 = round(float(thresh), 2)
            
        if fpr <= 0.005 and rec > best_rec_cs_05:
            best_rec_cs_05 = rec
            thresh_cs_05 = round(float(thresh), 2)
            
        if fpr <= 0.010 and rec > best_rec_cs_10:
            best_rec_cs_10 = rec
            thresh_cs_10 = round(float(thresh), 2)
            
    # Default to cost-sensitive balanced threshold (e.g. 0.10) for high generalization and >85% accuracy
    optimal_threshold = thresh_cs_05
    
    print(f"Validation Thresholds:")
    print(f"  - F1-Optimal Threshold: {optimal_threshold_f1:.2f} (Val F1: {best_f1:.4f})")
    print(f"  - Cost-Sensitive Balanced (Val FPR <= 0.5%): {thresh_cs_05:.2f} (Val Recall: {best_rec_cs_05:.4f})")
    print(f"  - Cost-Sensitive High-Detection (Val FPR <= 1.0%): {thresh_cs_10:.2f} (Val Recall: {best_rec_cs_10:.4f})")
    print(f"Selected Operating Threshold: {optimal_threshold:.2f}")
    
    # Save threshold curve data
    os.makedirs(REPORTS_DIR, exist_ok=True)
    thresh_path = os.path.join(REPORTS_DIR, 'threshold_optimization.json')
    with open(thresh_path, 'w') as f:
        json.dump({
            'optimal_threshold': optimal_threshold,
            'optimal_threshold_f1': optimal_threshold_f1,
            'threshold_cost_sensitive_balanced': thresh_cs_05,
            'threshold_cost_sensitive_high_detection': thresh_cs_10,
            'curve': records
        }, f, indent=2)
        
    return optimal_threshold, records

def compute_metrics(y_true, y_prob, threshold=0.5, train_time=0.0, inference_time=0.0, n_features=0):
    """
    Computes comprehensive evaluation metrics for intrusion detection.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'threshold': round(float(threshold), 4),
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'specificity': round(float(specificity), 4),
        'balanced_accuracy': round(float(balanced_acc), 4),
        'false_positive_rate': round(float(fpr), 4),
        'false_negative_rate': round(float(fnr), 4),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'training_time_seconds': round(float(train_time), 2),
        'inference_time_seconds': round(float(inference_time), 4),
        'number_of_features': int(n_features)
    }

def evaluate_model_comprehensive(model, X, y, threshold=0.5, train_time=0.0):
    """
    Evaluates a model, measuring inference time and returning full metric suite.
    """
    start_time = time.time()
    y_prob = model.predict_proba(X)[:, 1]
    inference_time = time.time() - start_time
    
    n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    metrics = compute_metrics(
        y_true=y,
        y_prob=y_prob,
        threshold=threshold,
        train_time=train_time,
        inference_time=inference_time,
        n_features=n_features
    )
    return metrics, y_prob

def evaluate_attack_categories(model, X, df_meta, threshold=0.5):
    """
    Evaluates detection performance broken down by attack category:
    Normal, DoS, Probe, R2L, U2R.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    df_eval = pd.DataFrame({
        'actual_category': df_meta['attack_category'],
        'actual_target': df_meta[TARGET_COLUMN],
        'pred_target': y_pred
    })
    
    category_results = {}
    for cat in ['normal', 'DoS', 'Probe', 'R2L', 'U2R']:
        subset = df_eval[df_eval['actual_category'] == cat]
        total = len(subset)
        if total == 0:
            continue
        if cat == 'normal':
            correct = int((subset['pred_target'] == 0).sum())
            rate = round(float(correct / total) * 100, 2)
            category_results[cat] = {
                'total_samples': total,
                'correctly_classified_normal': correct,
                'false_alarms': total - correct,
                'accuracy_pct': rate
            }
        else:
            detected = int((subset['pred_target'] == 1).sum())
            rate = round(float(detected / total) * 100, 2)
            category_results[cat] = {
                'total_samples': total,
                'detected_as_attack': detected,
                'missed_as_normal': total - detected,
                'detection_rate_pct': rate
            }
            
    return category_results

def evaluate_novel_attacks(model, X_test, df_test_meta, novel_attack_list, threshold=0.5):
    """
    Analyzes model detection performance specifically on unseen/novel attack types in KDDTest+.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    df_eval = pd.DataFrame({
        'attack_type': df_test_meta[LABEL_COLUMN],
        'attack_category': df_test_meta['attack_category'],
        'predicted_attack': y_pred
    })
    
    novel_results = []
    for atk in sorted(novel_attack_list):
        subset = df_eval[df_eval['attack_type'] == atk]
        total = len(subset)
        if total == 0:
            continue
        detected = int((subset['predicted_attack'] == 1).sum())
        missed = total - detected
        det_rate = round(float(detected / total) * 100, 2)
        category = subset['attack_category'].iloc[0]
        
        novel_results.append({
            'attack_type': atk,
            'attack_category': category,
            'total_samples': total,
            'detected_as_attack': detected,
            'missed_as_normal': missed,
            'detection_rate_pct': det_rate
        })
        
    os.makedirs(REPORTS_DIR, exist_ok=True)
    novel_path = os.path.join(REPORTS_DIR, 'novel_attack_evaluation.json')
    with open(novel_path, 'w') as f:
        json.dump(novel_results, f, indent=2)
        
    return novel_results
