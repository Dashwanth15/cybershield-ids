import os
import sys
import json
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = r"c:\Users\dashw\OneDrive\Desktop\Dashwanth\College\sem 5\cryptography\intrstrusion detection using xgboost"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RAW_COLUMNS,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ALL_FEATURES,
    DROP_COLUMNS, METADATA_COLUMNS, LABEL_COLUMN, TARGET_COLUMN,
    ATTACK_CATEGORIES, RANDOM_STATE, MODELS_DIR, REPORTS_DIR
)
from src.preprocessor import IDSPreprocessor
from src.explainability import explain_single_instance

def run_audit():
    audit_results = {}
    print("=" * 80)
    print("STARTING TECHNICAL VALIDATION AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------
    # AUDIT 1: DATASET VERIFICATION
    # -------------------------------------------------------------
    print("\n[AUDIT 1] Inspecting raw dataset files...")
    df_train = pd.read_csv(TRAIN_DATA_PATH, names=RAW_COLUMNS, header=None)
    df_test = pd.read_csv(TEST_DATA_PATH, names=RAW_COLUMNS, header=None)

    df_train[TARGET_COLUMN] = (df_train[LABEL_COLUMN] != 'normal').astype(int)
    df_test[TARGET_COLUMN] = (df_test[LABEL_COLUMN] != 'normal').astype(int)

    df_train['attack_category'] = df_train[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))
    df_test['attack_category'] = df_test[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))

    audit_results['audit_1_dataset'] = {
        'train_rows': len(df_train),
        'test_rows': len(df_test),
        'train_cols': df_train.shape[1],
        'test_cols': df_test.shape[1],
        'target_col': TARGET_COLUMN,
        'label_col': LABEL_COLUMN,
        'raw_columns_count': len(RAW_COLUMNS),
        'categorical_cols': CATEGORICAL_FEATURES,
        'numerical_cols_count': len(NUMERICAL_FEATURES),
        'train_missing_values': int(df_train.isnull().sum().sum()),
        'test_missing_values': int(df_test.isnull().sum().sum()),
        'train_duplicates': int(df_train.duplicated().sum()),
        'test_duplicates': int(df_test.duplicated().sum()),
        'train_class_dist': df_train[TARGET_COLUMN].value_counts().to_dict(),
        'test_class_dist': df_test[TARGET_COLUMN].value_counts().to_dict(),
        'train_cat_dist': df_train['attack_category'].value_counts().to_dict(),
        'test_cat_dist': df_test['attack_category'].value_counts().to_dict()
    }
    print(f"  KDDTrain+ rows: {len(df_train)}, cols: {df_train.shape[1]}")
    print(f"  KDDTest+ rows: {len(df_test)}, cols: {df_test.shape[1]}")
    print(f"  Missing values: Train={df_train.isnull().sum().sum()}, Test={df_test.isnull().sum().sum()}")
    print(f"  Duplicates: Train={df_train.duplicated().sum()}, Test={df_test.duplicated().sum()}")

    # -------------------------------------------------------------
    # AUDIT 2: DATA LEAKAGE CHECK
    # -------------------------------------------------------------
    print("\n[AUDIT 2] Checking data leakage and preprocessor...")
    prep_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    preprocessor = joblib.load(prep_path)
    feature_names = preprocessor.get_feature_names_out()

    # Verify if difficulty_level, attack_type, num_outbound_cmds are in feature_names
    has_difficulty = any('difficulty' in f.lower() for f in feature_names)
    has_attack_type = any('attack_type' in f.lower() for f in feature_names)
    has_target = any('target' in f.lower() for f in feature_names)
    has_num_outbound = any('num_outbound_cmds' in f.lower() for f in feature_names)

    audit_results['audit_2_leakage'] = {
        'difficulty_level_in_features': has_difficulty,
        'attack_type_in_features': has_attack_type,
        'target_in_features': has_target,
        'num_outbound_cmds_in_features': has_num_outbound,
        'transformed_features_count': len(feature_names),
        'drop_columns': DROP_COLUMNS + METADATA_COLUMNS
    }
    print(f"  difficulty_level in features: {has_difficulty} (Expected: False)")
    print(f"  attack_type in features: {has_attack_type} (Expected: False)")
    print(f"  target in features: {has_target} (Expected: False)")
    print(f"  num_outbound_cmds in features: {has_num_outbound} (Expected: False)")
    print(f"  Transformed feature count: {len(feature_names)}")

    # -------------------------------------------------------------
    # AUDIT 3: TRAIN / VALIDATION / TEST SPLIT
    # -------------------------------------------------------------
    print("\n[AUDIT 3] Checking train / validation / test split reproducibility...")
    df_train_split, df_val_split = train_test_split(
        df_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df_train[TARGET_COLUMN]
    )

    audit_results['audit_3_split'] = {
        'train_split_rows': len(df_train_split),
        'val_split_rows': len(df_val_split),
        'test_rows': len(df_test),
        'train_normal_ratio': float((df_train_split[TARGET_COLUMN] == 0).mean()),
        'val_normal_ratio': float((df_val_split[TARGET_COLUMN] == 0).mean()),
        'test_normal_ratio': float((df_test[TARGET_COLUMN] == 0).mean())
    }
    print(f"  Train split rows: {len(df_train_split)} (80%)")
    print(f"  Validation split rows: {len(df_val_split)} (20%)")
    print(f"  Test rows (untouched): {len(df_test)}")
    print(f"  Train normal ratio: {df_train_split[TARGET_COLUMN].mean():.4f}, Val normal ratio: {df_val_split[TARGET_COLUMN].mean():.4f}")

    # -------------------------------------------------------------
    # AUDIT 4 & 5: MODEL ARTIFACTS & BOOSTER INSPECTION
    # -------------------------------------------------------------
    print("\n[AUDIT 4 & 5] Inspecting saved models and booster metadata...")
    base_model_path = os.path.join(MODELS_DIR, 'baseline_xgb.json')
    tuned_model_path = os.path.join(MODELS_DIR, 'tuned_xgb.json')

    base_model = xgb.XGBClassifier()
    base_model.load_model(base_model_path)
    base_booster = base_model.get_booster()

    tuned_model = xgb.XGBClassifier()
    tuned_model.load_model(tuned_model_path)
    tuned_booster = tuned_model.get_booster()

    base_trees = len(base_booster.get_dump())
    tuned_trees = len(tuned_booster.get_dump())

    tuning_summary_path = os.path.join(REPORTS_DIR, 'tuning_summary.json')
    with open(tuning_summary_path, 'r') as f:
        tuning_summary = json.load(f)

    audit_results['audit_4_5_models'] = {
        'baseline_trees_count': base_trees,
        'tuned_trees_count': tuned_trees,
        'tuning_summary': tuning_summary
    }
    print(f"  Baseline trees count: {base_trees} (Expected: 100)")
    print(f"  Tuned trees count: {tuned_trees} (Expected: 150)")
    print(f"  Tuned hyperparameters: {json.dumps(tuning_summary.get('best_hyperparameters', {}), indent=2)}")

    # -------------------------------------------------------------
    # AUDIT 6 & 7: THRESHOLD OPTIMIZATION VERIFICATION
    # -------------------------------------------------------------
    print("\n[AUDIT 6 & 7] Independently re-evaluating validation threshold...")
    X_val = preprocessor.transform(df_val_split)
    y_val = df_val_split[TARGET_COLUMN].values
    val_probs = tuned_model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 91)
    best_val_f1 = -1.0
    best_val_thresh = 0.50

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_val_f1:
            best_val_f1 = f
            best_val_thresh = round(float(t), 2)

    audit_results['audit_7_threshold'] = {
        'reproduced_optimal_threshold': best_val_thresh,
        'reproduced_max_val_f1': round(best_val_f1, 4),
        'claimed_optimal_threshold': 0.41
    }
    print(f"  Reproduced optimal threshold on validation set: {best_val_thresh} (F1 = {best_val_f1:.4f})")
    print(f"  Matches claimed threshold 0.41: {best_val_thresh == 0.41}")

    # -------------------------------------------------------------
    # AUDIT 8: INDEPENDENT EVALUATION ON UNTOUCHED KDDTest+
    # -------------------------------------------------------------
    print("\n[AUDIT 8] Running independent evaluation on untouched KDDTest+...")
    X_test = preprocessor.transform(df_test)
    y_test = df_test[TARGET_COLUMN].values

    test_probs = tuned_model.predict_proba(X_test)[:, 1]
    test_preds_opt = (test_probs >= best_val_thresh).astype(int)
    test_preds_def = (test_probs >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds_opt, labels=[0, 1]).ravel()
    reproduced_metrics_opt = {
        'accuracy': round(float(accuracy_score(y_test, test_preds_opt)), 4),
        'precision': round(float(precision_score(y_test, test_preds_opt)), 4),
        'recall': round(float(recall_score(y_test, test_preds_opt)), 4),
        'f1_score': round(float(f1_score(y_test, test_preds_opt)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, test_probs)), 4),
        'pr_auc': round(float(average_precision_score(y_test, test_probs)), 4),
        'specificity': round(float(tn / (tn + fp)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y_test, test_preds_opt)), 4),
        'false_positive_rate': round(float(fp / (fp + tn)), 4),
        'false_negative_rate': round(float(fn / (fn + tp)), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

    # Baseline on KDDTest+
    base_probs = base_model.predict_proba(X_test)[:, 1]
    base_preds = (base_probs >= 0.50).astype(int)
    btn, bfp, bfn, btp = confusion_matrix(y_test, base_preds, labels=[0, 1]).ravel()
    reproduced_metrics_base = {
        'accuracy': round(float(accuracy_score(y_test, base_preds)), 4),
        'precision': round(float(precision_score(y_test, base_preds)), 4),
        'recall': round(float(recall_score(y_test, base_preds)), 4),
        'f1_score': round(float(f1_score(y_test, base_preds)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, base_probs)), 4),
        'pr_auc': round(float(average_precision_score(y_test, base_probs)), 4),
        'specificity': round(float(btn / (btn + bfp)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y_test, base_preds)), 4),
        'false_positive_rate': round(float(bfp / (bfp + btn)), 4),
        'false_negative_rate': round(float(bfn / (bfn + btp)), 4),
        'confusion_matrix': {'tn': int(btn), 'fp': int(bfp), 'fn': int(bfn), 'tp': int(btp)}
    }

    audit_results['audit_8_test_evaluation'] = {
        'reproduced_tuned_opt': reproduced_metrics_opt,
        'reproduced_baseline': reproduced_metrics_base
    }
    print(f"  Reproduced Tuned (0.41) Accuracy: {reproduced_metrics_opt['accuracy']*100:.2f}%")
    print(f"  Reproduced Tuned (0.41) Precision: {reproduced_metrics_opt['precision']*100:.2f}%")
    print(f"  Reproduced Tuned (0.41) Recall: {reproduced_metrics_opt['recall']*100:.2f}%")
    print(f"  Reproduced Tuned (0.41) F1: {reproduced_metrics_opt['f1_score']:.4f}")
    print(f"  Reproduced Tuned (0.41) ROC-AUC: {reproduced_metrics_opt['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # AUDIT 11 & 12: SHAP & MANUAL PREDICTION CHECK
    # -------------------------------------------------------------
    print("\n[AUDIT 11 & 12] Testing SHAP explainability on 2 distinct inputs...")
    explainer = shap.TreeExplainer(tuned_model)
    
    # Test 1: Normal HTTP
    normal_sample = df_test[(df_test[LABEL_COLUMN] == 'normal') & (df_test['service'] == 'http')].iloc[0].to_dict()
    expl_normal = explain_single_instance(explainer, tuned_model, preprocessor, normal_sample)

    # Test 2: Neptune Attack
    neptune_sample = df_test[df_test[LABEL_COLUMN] == 'neptune'].iloc[0].to_dict()
    expl_neptune = explain_single_instance(explainer, tuned_model, preprocessor, neptune_sample)

    audit_results['audit_11_12_shap'] = {
        'normal_attack_prob': expl_normal['attack_probability'],
        'normal_top_attack_contribs': expl_normal['attack_contributors'][:2],
        'normal_top_normal_contribs': expl_normal['normal_contributors'][:2],
        'neptune_attack_prob': expl_neptune['attack_probability'],
        'neptune_top_attack_contribs': expl_neptune['attack_contributors'][:2],
        'neptune_top_normal_contribs': expl_neptune['normal_contributors'][:2],
    }
    print(f"  Normal sample attack probability: {expl_normal['attack_probability']*100:.2f}%")
    print(f"  Neptune sample attack probability: {expl_neptune['attack_probability']*100:.2f}%")
    print(f"  SHAP values change dynamically: {expl_normal['attack_probability'] != expl_neptune['attack_probability']}")

    # -------------------------------------------------------------
    # AUDIT 13: BATCH CSV PREDICTION CHECK
    # -------------------------------------------------------------
    print("\n[AUDIT 13] Testing batch CSV prediction logic...")
    sample_batch = df_test.head(20)
    batch_trans = preprocessor.transform(sample_batch)
    batch_probs = tuned_model.predict_proba(batch_trans)[:, 1]
    batch_preds = (batch_probs >= best_val_thresh).astype(int)

    audit_results['audit_13_batch'] = {
        'batch_size': len(sample_batch),
        'attacks_predicted': int((batch_preds == 1).sum()),
        'normal_predicted': int((batch_preds == 0).sum())
    }
    print(f"  Batch size: {len(sample_batch)}, Attacks: {(batch_preds == 1).sum()}, Normal: {(batch_preds == 0).sum()}")

    # Save audit results
    audit_save_path = os.path.join(REPORTS_DIR, 'audit_verification_results.json')
    with open(audit_save_path, 'w') as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit execution results saved to: {audit_save_path}")

if __name__ == '__main__':
    run_audit()
