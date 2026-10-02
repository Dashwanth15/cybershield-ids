import os
import sys
import json
import time
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_COLUMN,
    REPORTS_DIR, MODELS_DIR, FIGURES_DIR, ALL_FEATURES
)
from src.data_loader import load_raw_data, get_train_val_split, extract_real_presets
from src.preprocessor import IDSPreprocessor
from src.eda import run_eda
from src.train import train_baseline_model, tune_xgboost_model
from src.evaluate import (
    optimize_threshold, evaluate_model_comprehensive,
    evaluate_attack_categories, evaluate_novel_attacks
)
from src.explainability import get_feature_importances, compute_shap_global
from src.visualize import (
    plot_class_and_category_distribution, plot_correlation_matrix,
    plot_confusion_matrices, plot_roc_curves, plot_precision_recall_curves,
    plot_threshold_optimization, plot_feature_importance_and_shap,
    plot_model_comparison_barchart
)

def main():
    print("=" * 80)
    print(" MACHINE LEARNING-BASED INTRUSION DETECTION SYSTEM (IDS)")
    print(" MODEL: XGBOOST | DATASET: NSL-KDD BENCHMARK")
    print("=" * 80)

    # 1. Load Data
    df_train, df_test = load_raw_data(TRAIN_DATA_PATH, TEST_DATA_PATH)

    # 2. Exploratory Data Analysis
    eda_summary = run_eda(df_train, df_test)

    # 3. Train/Validation Split (80% Train, 20% Validation)
    df_train_split, df_val_split = get_train_val_split(df_train, val_size=0.20)

    # 4. Fit Preprocessor Strictly on Training Partition
    preprocessor = IDSPreprocessor()
    preprocessor.fit(df_train_split)
    preprocessor.save()

    # 6. Transform Features
    print("\nTransforming feature matrices...")
    X_train = preprocessor.transform(df_train_split)
    y_train = df_train_split[TARGET_COLUMN].values

    X_val = preprocessor.transform(df_val_split)
    y_val = df_val_split[TARGET_COLUMN].values

    X_test = preprocessor.transform(df_test)
    y_test = df_test[TARGET_COLUMN].values

    feature_names = preprocessor.get_feature_names_out()
    print(f"Feature matrix dimensions: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # 7. Train Baseline XGBoost Model
    baseline_model, base_train_time = train_baseline_model(X_train, y_train)

    # 8. Evaluate Baseline on Validation & Untouched Test Set
    base_val_metrics, base_val_prob = evaluate_model_comprehensive(baseline_model, X_val, y_val, threshold=0.5, train_time=base_train_time)
    base_test_metrics, base_test_prob = evaluate_model_comprehensive(baseline_model, X_test, y_test, threshold=0.5, train_time=base_train_time)

    print(f"\nBaseline Model Test Accuracy: {base_test_metrics['accuracy']:.4f} | F1: {base_test_metrics['f1_score']:.4f} | ROC-AUC: {base_test_metrics['roc_auc']:.4f}")

    # 9. Hyperparameter Tuning (RandomizedSearchCV with 5-Fold Stratified CV)
    tuned_model, best_params, tuning_time = tune_xgboost_model(X_train, y_train, cv=5, n_iter=15)

    # 10. Classification Threshold Optimization on Validation Data Only
    optimal_thresh, thresh_records = optimize_threshold(tuned_model, X_val, y_val)

    # 11. Evaluate Tuned Model on Validation & Untouched Test Set
    tuned_val_metrics_def, tuned_val_prob = evaluate_model_comprehensive(tuned_model, X_val, y_val, threshold=0.5, train_time=tuning_time)
    tuned_val_metrics_opt, _ = evaluate_model_comprehensive(tuned_model, X_val, y_val, threshold=optimal_thresh, train_time=tuning_time)

    # Final evaluation on untouched KDDTest+ at default (0.50) and optimal threshold
    tuned_test_metrics_def, tuned_test_prob = evaluate_model_comprehensive(tuned_model, X_test, y_test, threshold=0.5, train_time=tuning_time)
    tuned_test_metrics_opt, _ = evaluate_model_comprehensive(tuned_model, X_test, y_test, threshold=optimal_thresh, train_time=tuning_time)

    print(f"\nOptimized Model Test (Threshold 0.50) Accuracy: {tuned_test_metrics_def['accuracy']:.4f} | F1: {tuned_test_metrics_def['f1_score']:.4f} | ROC-AUC: {tuned_test_metrics_def['roc_auc']:.4f}")
    print(f"Optimized Model Test (Threshold {optimal_thresh:.2f}) Accuracy: {tuned_test_metrics_opt['accuracy']:.4f} | F1: {tuned_test_metrics_opt['f1_score']:.4f}")

    # 12. Attack Category Breakdown
    cat_eval = evaluate_attack_categories(tuned_model, X_test, df_test, threshold=optimal_thresh)
    print("\nAttack Category Performance on KDDTest+:")
    for cat, info in cat_eval.items():
        print(f"  {cat.upper():<8}: {json.dumps(info)}")

    # 13. Novel / Unseen Attack Types Evaluation
    novel_attacks = eda_summary['novel_attack_types_in_test']
    novel_eval = evaluate_novel_attacks(tuned_model, X_test, df_test, novel_attacks, threshold=optimal_thresh)
    print(f"\nEvaluated {len(novel_eval)} novel attack types in KDDTest+.")

    # 14. Extract Real Test Presets for Web App (Strictly AFTER final evaluation)
    presets = extract_real_presets(df_test)

    # 15. Feature Importance & SHAP
    print("\nExtracting feature importances and computing SHAP values...")
    feat_importance = get_feature_importances(tuned_model, feature_names, top_n=20)
    
    # Use a representative sample (1000 records) of test data for SHAP computation
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    X_shap_sample = X_test[sample_idx]
    explainer, shap_values, top_shap = compute_shap_global(tuned_model, X_shap_sample, feature_names)

    # 15. Generate Visualizations
    print("\nGenerating publication-grade figures...")
    plot_class_and_category_distribution(df_train, df_test)
    plot_correlation_matrix(df_train)
    plot_confusion_matrices(y_test, base_test_prob, tuned_test_prob, thresh_base=0.5, thresh_tuned=optimal_thresh)
    plot_roc_curves(y_test, base_test_prob, tuned_test_prob, base_test_metrics['roc_auc'], tuned_test_metrics_opt['roc_auc'])
    plot_precision_recall_curves(y_test, base_test_prob, tuned_test_prob, base_test_metrics['pr_auc'], tuned_test_metrics_opt['pr_auc'])
    plot_threshold_optimization(thresh_records, optimal_thresh)
    plot_feature_importance_and_shap(feat_importance, shap_values, feature_names)
    plot_model_comparison_barchart(base_test_metrics, tuned_test_metrics_opt)

    # 16. Save Master Results
    master_results = {
        'dataset': {
            'name': 'NSL-KDD',
            'train_samples': len(df_train),
            'test_samples': len(df_test),
            'active_features_count': len(ALL_FEATURES),
            'encoded_features_count': len(feature_names),
            'classes': ['Normal (0)', 'Attack (1)']
        },
        'best_hyperparameters': best_params,
        'optimal_threshold': optimal_thresh,
        'validation_metrics': {
            'baseline': base_val_metrics,
            'tuned_threshold_0_50': tuned_val_metrics_def,
            'tuned_threshold_optimal': tuned_val_metrics_opt
        },
        'test_metrics': {
            'baseline': base_test_metrics,
            'tuned_threshold_0_50': tuned_test_metrics_def,
            'tuned_threshold_optimal': tuned_test_metrics_opt
        },
        'attack_category_performance': cat_eval,
        'novel_attacks_performance': novel_eval
    }

    results_path = os.path.join(REPORTS_DIR, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"\nFinal master results saved to {results_path}")

    # Generate Model Comparison Markdown Table
    comparison_md = f"""# Model Performance Comparison on Untouched KDDTest+

| Metric | Baseline XGBoost (0.50) | Tuned XGBoost (0.50) | Tuned XGBoost ({optimal_thresh:.2f}) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | {base_test_metrics['accuracy']*100:.2f}% | {tuned_test_metrics_def['accuracy']*100:.2f}% | {tuned_test_metrics_opt['accuracy']*100:.2f}% |
| **Precision** | {base_test_metrics['precision']*100:.2f}% | {tuned_test_metrics_def['precision']*100:.2f}% | {tuned_test_metrics_opt['precision']*100:.2f}% |
| **Recall (Detection Rate)** | {base_test_metrics['recall']*100:.2f}% | {tuned_test_metrics_def['recall']*100:.2f}% | {tuned_test_metrics_opt['recall']*100:.2f}% |
| **F1-Score** | {base_test_metrics['f1_score']:.4f} | {tuned_test_metrics_def['f1_score']:.4f} | {tuned_test_metrics_opt['f1_score']:.4f} |
| **ROC-AUC** | {base_test_metrics['roc_auc']:.4f} | {tuned_test_metrics_def['roc_auc']:.4f} | {tuned_test_metrics_opt['roc_auc']:.4f} |
| **PR-AUC** | {base_test_metrics['pr_auc']:.4f} | {tuned_test_metrics_def['pr_auc']:.4f} | {tuned_test_metrics_opt['pr_auc']:.4f} |
| **Specificity** | {base_test_metrics['specificity']*100:.2f}% | {tuned_test_metrics_def['specificity']*100:.2f}% | {tuned_test_metrics_opt['specificity']*100:.2f}% |
| **Balanced Accuracy** | {base_test_metrics['balanced_accuracy']*100:.2f}% | {tuned_test_metrics_def['balanced_accuracy']*100:.2f}% | {tuned_test_metrics_opt['balanced_accuracy']*100:.2f}% |
| **False Positive Rate (FPR)** | {base_test_metrics['false_positive_rate']*100:.2f}% | {tuned_test_metrics_def['false_positive_rate']*100:.2f}% | {tuned_test_metrics_opt['false_positive_rate']*100:.2f}% |
| **False Negative Rate (FNR)** | {base_test_metrics['false_negative_rate']*100:.2f}% | {tuned_test_metrics_def['false_negative_rate']*100:.2f}% | {tuned_test_metrics_opt['false_negative_rate']*100:.2f}% |
| **Inference Time (22,544 rows)** | {base_test_metrics['inference_time_seconds']:.4f} s | {tuned_test_metrics_def['inference_time_seconds']:.4f} s | {tuned_test_metrics_opt['inference_time_seconds']:.4f} s |
"""
    with open(os.path.join(REPORTS_DIR, 'model_comparison.md'), 'w') as f:
        f.write(comparison_md)

    print("\n" + "=" * 80)
    print(" ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
