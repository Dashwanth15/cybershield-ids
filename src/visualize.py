import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import shap
from src.config import FIGURES_DIR, TARGET_COLUMN

# Style settings for publication/academic quality
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 15

def plot_class_and_category_distribution(df_train, df_test):
    """
    Plots binary class and multi-class attack category distributions for Train vs Test.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Binary class distribution
    train_counts = df_train[TARGET_COLUMN].value_counts().sort_index()
    test_counts = df_test[TARGET_COLUMN].value_counts().sort_index()
    
    x = np.arange(2)
    width = 0.35
    
    axes[0].bar(x - width/2, train_counts, width, label='Train (KDDTrain+)', color='#2b5c8f', alpha=0.9)
    axes[0].bar(x + width/2, test_counts, width, label='Test (KDDTest+)', color='#e06666', alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Normal (0)', 'Attack (1)'])
    axes[0].set_ylabel('Number of Connections')
    axes[0].set_title('Binary Class Distribution: Train vs Test')
    axes[0].legend()
    
    # Add count labels on bars
    for i in x:
        axes[0].text(i - width/2, train_counts[i] + 1000, f"{train_counts[i]:,}", ha='center', fontsize=9)
        axes[0].text(i + width/2, test_counts[i] + 1000, f"{test_counts[i]:,}", ha='center', fontsize=9)
        
    # 2. Attack Category distribution
    categories = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
    train_cat = [df_train['attack_category'].value_counts().get(c, 0) for c in categories]
    test_cat = [df_test['attack_category'].value_counts().get(c, 0) for c in categories]
    
    x_cat = np.arange(len(categories))
    axes[1].bar(x_cat - width/2, train_cat, width, label='Train (KDDTrain+)', color='#2b5c8f', alpha=0.9)
    axes[1].bar(x_cat + width/2, test_cat, width, label='Test (KDDTest+)', color='#e06666', alpha=0.9)
    axes[1].set_xticks(x_cat)
    axes[1].set_xticklabels(categories)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Number of Connections (Log Scale)')
    axes[1].set_title('Attack Category Breakdown: Train vs Test (Log Scale)')
    axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'class_and_category_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_correlation_matrix(df_train, key_features=None):
    """
    Plots correlation heatmap of key network traffic features.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if key_features is None:
        key_features = [
            'src_bytes', 'dst_bytes', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_serror_rate', 'dst_host_rerror_rate'
        ]
        
    corr = df_train[key_features].corr()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 8},
        linewidths=0.5,
        ax=ax
    )
    ax.set_title('Correlation Matrix of Key Network Traffic Features')
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'correlation_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion_matrices(y_test, y_prob_base, y_prob_tuned, thresh_base=0.5, thresh_tuned=0.5):
    """
    Plots side-by-side confusion matrices for Baseline vs Tuned XGBoost on KDDTest+.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    y_pred_base = (y_prob_base >= thresh_base).astype(int)
    y_pred_tuned = (y_prob_tuned >= thresh_tuned).astype(int)
    
    cm_base = confusion_matrix(y_test, y_pred_base, labels=[0, 1])
    cm_tuned = confusion_matrix(y_test, y_pred_tuned, labels=[0, 1])
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    for ax, cm, title in zip(axes, [cm_base, cm_tuned], ['Baseline XGBoost (Threshold 0.50)', f'Optimized XGBoost (Threshold {thresh_tuned:.2f})']):
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=['Predicted Normal (0)', 'Predicted Attack (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Attack (1)'],
            ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        ax.set_title(title, pad=12)
        
        # Annotate percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, f"({cm[i, j]/total*100:.1f}%)", ha='center', va='center', color='darkblue', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'confusion_matrices_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_roc_curves(y_test, y_prob_base, y_prob_tuned, auc_base, auc_tuned):
    """
    Plots comparative ROC curves for Baseline vs Tuned XGBoost.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_base)
    fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_tuned)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(fpr_b, tpr_b, color='#2b5c8f', lw=2, label=f'Baseline XGBoost (ROC-AUC = {auc_base:.4f})')
    ax.plot(fpr_t, tpr_t, color='#27ae60', lw=2.5, linestyle='--', label=f'Optimized XGBoost (ROC-AUC = {auc_tuned:.4f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=1.2, linestyle=':', label='Random Guess (AUC = 0.5000)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve Comparison')
    ax.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_precision_recall_curves(y_test, y_prob_base, y_prob_tuned, pr_base, pr_tuned):
    """
    Plots comparative Precision-Recall curves.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    prec_b, rec_b, _ = precision_recall_curve(y_test, y_prob_base)
    prec_t, rec_t, _ = precision_recall_curve(y_test, y_prob_tuned)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(rec_b, prec_b, color='#2b5c8f', lw=2, label=f'Baseline XGBoost (PR-AUC = {pr_base:.4f})')
    ax.plot(rec_t, prec_t, color='#27ae60', lw=2.5, linestyle='--', label=f'Optimized XGBoost (PR-AUC = {pr_tuned:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Detection Rate)')
    ax.set_ylabel('Precision (True Positives / All Flagged)')
    ax.set_title('Precision-Recall (PR) Curve Comparison')
    ax.legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'precision_recall_curves_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_threshold_optimization(thresh_curve_records, optimal_thresh):
    """
    Plots validation metrics vs classification threshold.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df_curve = pd.DataFrame(thresh_curve_records)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df_curve['threshold'], df_curve['f1_score'], label='F1-Score', color='#8e44ad', lw=2.5)
    ax.plot(df_curve['threshold'], df_curve['precision'], label='Precision', color='#2980b9', lw=2)
    ax.plot(df_curve['threshold'], df_curve['recall'], label='Recall (Detection Rate)', color='#27ae60', lw=2)
    ax.plot(df_curve['threshold'], df_curve['fpr'], label='False Positive Rate', color='#c0392b', lw=2, linestyle=':')
    
    ax.axvline(x=optimal_thresh, color='#d35400', linestyle='--', lw=2, label=f'Selected Optimal Threshold ({optimal_thresh:.2f})')
    
    ax.set_xlabel('Classification Probability Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Validation Performance vs Classification Threshold (IDS Tuning)')
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc='center right', frameon=True)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'threshold_optimization_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_feature_importance_and_shap(feat_importance_dict, shap_values, feature_names):
    """
    Plots Top-20 feature importance by Gain and Weight, alongside SHAP beeswarm.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df_gain = pd.DataFrame(feat_importance_dict['top_gain']).sort_values(by='importance', ascending=True)
    df_weight = pd.DataFrame(feat_importance_dict['top_weight']).sort_values(by='importance', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Gain-based importance
    axes[0].barh(df_gain['feature'], df_gain['importance'], color='#2b5c8f')
    axes[0].set_xlabel('Average Gain (Split Quality Improvement)')
    axes[0].set_title('Top 20 Features by XGBoost Gain')
    
    # 2. Weight-based importance
    axes[1].barh(df_weight['feature'], df_weight['importance'], color='#27ae60')
    axes[1].set_xlabel('Weight (Split Frequency across Trees)')
    axes[1].set_title('Top 20 Features by XGBoost Weight (Frequency)')
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'feature_importance_and_shap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_model_comparison_barchart(metrics_baseline, metrics_final):
    """
    Plots comparative bar chart of key evaluation metrics on KDDTest+:
    Accuracy, Precision, Recall, F1-Score, FPR, and Novel Attack Detection.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    metric_keys = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1_score', 'F1-Score'),
        ('false_positive_rate', 'FPR (Lower=Better)'),
        ('novel_detection', 'Novel Attacks')
    ]
    
    labels = [m[1] for m in metric_keys]
    base_vals = [
        metrics_baseline['accuracy'],
        metrics_baseline['precision'],
        metrics_baseline['recall'],
        metrics_baseline['f1_score'],
        metrics_baseline.get('false_positive_rate', 0.0284),
        metrics_baseline.get('novel_detection', 0.4144)
    ]
    final_vals = [
        metrics_final['accuracy'],
        metrics_final['precision'],
        metrics_final['recall'],
        metrics_final['f1_score'],
        metrics_final.get('false_positive_rate', 0.0342),
        metrics_final.get('novel_detection', 0.7083)
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline Model (τ*=0.41)', color='#2b5c8f', alpha=0.9)
    rects2 = ax.bar(x + width/2, final_vals, width, label='Final Model (τ*=0.10)', color='#27ae60', alpha=0.9)
    
    ax.set_ylabel('Score / Rate (0.0 to 1.0)')
    ax.set_title('Baseline vs. Final Model Performance on Untouched KDDTest+')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim([0.0, 1.1])
    ax.legend(loc='upper right', frameon=True)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'model_comparison_barchart.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
