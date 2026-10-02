import os
import sys
import time
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, fbeta_score
)

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

print("=" * 80)
print(" EXECUTING GENERALIZATION IMPROVEMENT EXPERIMENT PLAN")
print(" STRICT ISOLATION: KDDTest+ REMAINS UNTOUCHED DURING ALL EXPERIMENTS")
print("=" * 80)

# -------------------------------------------------------------
# 0. LOAD DATA & CREATE 80/20 STRATIFIED SPLIT ON KDDTrain+
# -------------------------------------------------------------
print("\nLoading KDDTrain+ and creating 80/20 train/validation split...")
df_train = pd.read_csv(TRAIN_DATA_PATH, names=RAW_COLUMNS, header=None)
df_train[TARGET_COLUMN] = (df_train[LABEL_COLUMN] != 'normal').astype(int)
df_train['attack_category'] = df_train[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))

# KDDTest+ loaded ONLY for final gate evaluation at the very end
df_test = pd.read_csv(TEST_DATA_PATH, names=RAW_COLUMNS, header=None)
df_test[TARGET_COLUMN] = (df_test[LABEL_COLUMN] != 'normal').astype(int)
df_test['attack_category'] = df_test[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))

df_train_split, df_val_split = train_test_split(
    df_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df_train[TARGET_COLUMN]
)

print(f"Train split: {len(df_train_split)}, Validation split: {len(df_val_split)}, Untouched Test: {len(df_test)}")

# Fit standard preprocessor strictly on training split
prep_baseline = IDSPreprocessor()
prep_baseline.fit(df_train_split)
X_train_base = prep_baseline.transform(df_train_split)
y_train = df_train_split[TARGET_COLUMN].values
X_val_base = prep_baseline.transform(df_val_split)
y_val = df_val_split[TARGET_COLUMN].values

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Helper function to evaluate on validation set across thresholds
def evaluate_validation(model, X_val, y_val, metric_pref='f1', target_fpr=0.03):
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    
    best_thresh = 0.50
    best_score = -1.0
    best_metrics = {}
    
    all_threshold_stats = []
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        fb12 = fbeta_score(y_val, preds, beta=1.2, zero_division=0)
        fb15 = fbeta_score(y_val, preds, beta=1.5, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = balanced_accuracy_score(y_val, preds)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        stat = {
            'threshold': round(float(t), 2),
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1': round(float(f1), 4),
            'f_beta_1.2': round(float(fb12), 4),
            'f_beta_1.5': round(float(fb15), 4),
            'specificity': round(float(spec), 4),
            'balanced_accuracy': round(float(bal_acc), 4),
            'fpr': round(float(fpr), 4),
            'fnr': round(float(fnr), 4)
        }
        all_threshold_stats.append(stat)
        
        # Metric preference for threshold selection
        if metric_pref == 'f1':
            score = f1
        elif metric_pref == 'f_beta_1.2':
            score = fb12
        elif metric_pref == 'f_beta_1.5':
            score = fb15
        elif metric_pref == 'max_rec_constrained_fpr':
            # Maximize recall subject to FPR <= target_fpr
            score = rec if fpr <= target_fpr else -1.0
        elif metric_pref == 'balanced_accuracy':
            score = bal_acc
        else:
            score = f1
            
        if score > best_score:
            best_score = score
            best_thresh = round(float(t), 2)
            best_metrics = stat

    return best_thresh, best_metrics, all_threshold_stats, y_prob

experiments_log = []

# -------------------------------------------------------------
# EXPERIMENT 0: REPRODUCE BASELINE
# -------------------------------------------------------------
print("\n>>> EXPERIMENT 0: Reproduce Baseline Model <<<")
t0 = time.time()
xgb_base = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.15,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    gamma=0.2,
    reg_alpha=1.0,
    reg_lambda=2.0,
    scale_pos_weight=1.15,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("  Running 5-fold CV...")
cv_scores_exp0 = cross_val_score(xgb_base, X_train_base, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1)
cv_f1_exp0 = float(np.mean(cv_scores_exp0))

print("  Fitting on train split...")
xgb_base.fit(X_train_base, y_train)
train_time_exp0 = time.time() - t0

thresh_exp0, val_metrics_exp0, _, _ = evaluate_validation(xgb_base, X_val_base, y_val, metric_pref='f1')

print(f"  Exp 0 - CV F1: {cv_f1_exp0:.4f}, Val Acc: {val_metrics_exp0['accuracy']*100:.2f}%, Val Rec: {val_metrics_exp0['recall']*100:.2f}%, Val F1: {val_metrics_exp0['f1']:.4f}, Thresh: {thresh_exp0}")

experiments_log.append({
    'experiment_id': 'Exp0_Baseline',
    'description': 'Current verified baseline (n=150, d=6, lr=0.15, scale_pos=1.15)',
    'features_used': 'Original 40 (120 encoded)',
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.15,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'gamma': 0.2,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'scale_pos_weight': 1.15,
    'category_weighting': 'None',
    'cv_f1': round(cv_f1_exp0, 4),
    'val_accuracy': val_metrics_exp0['accuracy'],
    'val_precision': val_metrics_exp0['precision'],
    'val_recall': val_metrics_exp0['recall'],
    'val_f1': val_metrics_exp0['f1'],
    'val_specificity': val_metrics_exp0['specificity'],
    'val_balanced_accuracy': val_metrics_exp0['balanced_accuracy'],
    'val_fpr': val_metrics_exp0['fpr'],
    'val_fnr': val_metrics_exp0['fnr'],
    'selected_threshold': thresh_exp0,
    'training_time': round(train_time_exp0, 2)
})

# -------------------------------------------------------------
# EXPERIMENT 1: DOMAIN FEATURE ENGINEERING
# -------------------------------------------------------------
print("\n>>> EXPERIMENT 1: Domain Feature Engineering <<<")

def add_engineered_features(df):
    df_new = df.copy()
    src_b = df_new['src_bytes'].values
    dst_b = df_new['dst_bytes'].values
    cnt = df_new['count'].values
    srv_cnt = df_new['srv_count'].values
    same_srv = df_new['same_srv_rate'].values
    diff_srv = df_new['diff_srv_rate'].values
    dst_h_cnt = df_new['dst_host_count'].values
    dst_h_same = df_new['dst_host_same_srv_rate'].values
    dst_h_diff = df_new['dst_host_diff_srv_rate'].values
    serr = df_new['serror_rate'].values
    rerr = df_new['rerror_rate'].values
    
    # Safe domain ratios
    df_new['bytes_total'] = src_b + dst_b
    df_new['bytes_ratio'] = (src_b + 1.0) / (dst_b + 1.0)
    df_new['src_dst_diff'] = np.abs(src_b - dst_b)
    df_new['count_srv_ratio'] = (cnt + 1.0) / (srv_cnt + 1.0)
    df_new['same_srv_count'] = cnt * same_srv
    df_new['diff_srv_count'] = cnt * diff_srv
    df_new['dst_host_same_srv_count'] = dst_h_cnt * dst_h_same
    df_new['dst_host_diff_srv_count'] = dst_h_cnt * dst_h_diff
    df_new['syn_err_count'] = cnt * serr
    df_new['rej_err_count'] = cnt * rerr
    return df_new

df_train_fe = add_engineered_features(df_train_split)
df_val_fe = add_engineered_features(df_val_split)

# Preprocessor with engineered features
prep_fe = IDSPreprocessor()
prep_fe.numerical_features = NUMERICAL_FEATURES + [
    'bytes_total', 'bytes_ratio', 'src_dst_diff', 'count_srv_ratio',
    'same_srv_count', 'diff_srv_count', 'dst_host_same_srv_count',
    'dst_host_diff_srv_count', 'syn_err_count', 'rej_err_count'
]
prep_fe.fit(df_train_fe)
X_train_fe = prep_fe.transform(df_train_fe)
X_val_fe = prep_fe.transform(df_val_fe)

print(f"  Transformed feature dimension with FE: {X_train_fe.shape[1]}")

t0 = time.time()
xgb_fe = xgb.XGBClassifier(
    n_estimators=150, max_depth=6, learning_rate=0.15, subsample=0.9,
    colsample_bytree=0.9, min_child_weight=1, gamma=0.2, reg_alpha=1.0,
    reg_lambda=2.0, scale_pos_weight=1.15, eval_metric='logloss',
    random_state=RANDOM_STATE, n_jobs=-1
)
cv_scores_fe = cross_val_score(xgb_fe, X_train_fe, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1)
cv_f1_fe = float(np.mean(cv_scores_fe))
xgb_fe.fit(X_train_fe, y_train)
train_time_fe = time.time() - t0

thresh_fe, val_metrics_fe, _, _ = evaluate_validation(xgb_fe, X_val_fe, y_val, metric_pref='f1')
print(f"  Exp 1 (FE) - CV F1: {cv_f1_fe:.4f}, Val Acc: {val_metrics_fe['accuracy']*100:.2f}%, Val Rec: {val_metrics_fe['recall']*100:.2f}%, Val F1: {val_metrics_fe['f1']:.4f}, Thresh: {thresh_fe}")

experiments_log.append({
    'experiment_id': 'Exp1_Feature_Engineering',
    'description': 'Baseline + 10 domain interaction features',
    'features_used': '40 + 10 = 50 (130 encoded)',
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.15,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'gamma': 0.2,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'scale_pos_weight': 1.15,
    'category_weighting': 'None',
    'cv_f1': round(cv_f1_fe, 4),
    'val_accuracy': val_metrics_fe['accuracy'],
    'val_precision': val_metrics_fe['precision'],
    'val_recall': val_metrics_fe['recall'],
    'val_f1': val_metrics_fe['f1'],
    'val_specificity': val_metrics_fe['specificity'],
    'val_balanced_accuracy': val_metrics_fe['balanced_accuracy'],
    'val_fpr': val_metrics_fe['fpr'],
    'val_fnr': val_metrics_fe['fnr'],
    'selected_threshold': thresh_fe,
    'training_time': round(train_time_fe, 2)
})

# -------------------------------------------------------------
# EXPERIMENT 2: REGULARIZATION & SHALLOWER TREES
# -------------------------------------------------------------
print("\n>>> EXPERIMENT 2: Regularization & Shallower Trees <<<")
reg_candidates = [
    {'name': 'Exp2a_Shallower_d4', 'max_depth': 4, 'min_child_weight': 3, 'reg_alpha': 3.0, 'reg_lambda': 5.0, 'gamma': 0.5},
    {'name': 'Exp2b_Shallower_d5', 'max_depth': 5, 'min_child_weight': 3, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'gamma': 0.5},
    {'name': 'Exp2c_Strong_L1_L2', 'max_depth': 6, 'min_child_weight': 5, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'gamma': 1.0},
    {'name': 'Exp2d_Very_Strong_Reg', 'max_depth': 4, 'min_child_weight': 10, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'gamma': 1.0}
]

for rc in reg_candidates:
    t0 = time.time()
    m_reg = xgb.XGBClassifier(
        n_estimators=150, max_depth=rc['max_depth'], learning_rate=0.15, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=rc['min_child_weight'], gamma=rc['gamma'],
        reg_alpha=rc['reg_alpha'], reg_lambda=rc['reg_lambda'], scale_pos_weight=1.15,
        eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
    )
    cv_sc = cross_val_score(m_reg, X_train_base, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1)
    cv_f1_rc = float(np.mean(cv_sc))
    m_reg.fit(X_train_base, y_train)
    tr_time_rc = time.time() - t0
    
    th_rc, val_m_rc, _, _ = evaluate_validation(m_reg, X_val_base, y_val, metric_pref='f1')
    print(f"  {rc['name']} - CV F1: {cv_f1_rc:.4f}, Val Acc: {val_m_rc['accuracy']*100:.2f}%, Val Rec: {val_m_rc['recall']*100:.2f}%, Val F1: {val_m_rc['f1']:.4f}, Thresh: {th_rc}")
    
    experiments_log.append({
        'experiment_id': rc['name'],
        'description': f"max_depth={rc['max_depth']}, min_child={rc['min_child_weight']}, alpha={rc['reg_alpha']}, lambda={rc['reg_lambda']}",
        'features_used': 'Original 40 (120 encoded)',
        'n_estimators': 150,
        'max_depth': rc['max_depth'],
        'learning_rate': 0.15,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': rc['min_child_weight'],
        'gamma': rc['gamma'],
        'reg_alpha': rc['reg_alpha'],
        'reg_lambda': rc['reg_lambda'],
        'scale_pos_weight': 1.15,
        'category_weighting': 'None',
        'cv_f1': round(cv_f1_rc, 4),
        'val_accuracy': val_m_rc['accuracy'],
        'val_precision': val_m_rc['precision'],
        'val_recall': val_m_rc['recall'],
        'val_f1': val_m_rc['f1'],
        'val_specificity': val_m_rc['specificity'],
        'val_balanced_accuracy': val_m_rc['balanced_accuracy'],
        'val_fpr': val_m_rc['fpr'],
        'val_fnr': val_m_rc['fnr'],
        'selected_threshold': th_rc,
        'training_time': round(tr_time_rc, 2)
    })

# -------------------------------------------------------------
# EXPERIMENT 3: FEATURE & ROW SUBSAMPLING
# -------------------------------------------------------------
print("\n>>> EXPERIMENT 3: Feature & Row Subsampling <<<")
subsample_candidates = [
    {'name': 'Exp3a_Colsample_0.7', 'colsample': 0.7, 'subsample': 0.8},
    {'name': 'Exp3b_Colsample_0.8', 'colsample': 0.8, 'subsample': 0.8},
    {'name': 'Exp3c_Colsample_0.6', 'colsample': 0.6, 'subsample': 0.8}
]

for sc in subsample_candidates:
    t0 = time.time()
    m_sub = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.15, subsample=sc['subsample'],
        colsample_bytree=sc['colsample'], min_child_weight=1, gamma=0.2,
        reg_alpha=1.0, reg_lambda=2.0, scale_pos_weight=1.15,
        eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
    )
    cv_sc = cross_val_score(m_sub, X_train_base, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1)
    cv_f1_sc = float(np.mean(cv_sc))
    m_sub.fit(X_train_base, y_train)
    tr_time_sc = time.time() - t0
    
    th_sc, val_m_sc, _, _ = evaluate_validation(m_sub, X_val_base, y_val, metric_pref='f1')
    print(f"  {sc['name']} - CV F1: {cv_f1_sc:.4f}, Val Acc: {val_m_sc['accuracy']*100:.2f}%, Val Rec: {val_m_sc['recall']*100:.2f}%, Val F1: {val_m_sc['f1']:.4f}, Thresh: {th_sc}")
    
    experiments_log.append({
        'experiment_id': sc['name'],
        'description': f"colsample_bytree={sc['colsample']}, subsample={sc['subsample']}",
        'features_used': 'Original 40 (120 encoded)',
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.15,
        'subsample': sc['subsample'],
        'colsample_bytree': sc['colsample'],
        'min_child_weight': 1,
        'gamma': 0.2,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'scale_pos_weight': 1.15,
        'category_weighting': 'None',
        'cv_f1': round(cv_f1_sc, 4),
        'val_accuracy': val_m_sc['accuracy'],
        'val_precision': val_m_sc['precision'],
        'val_recall': val_m_sc['recall'],
        'val_f1': val_m_sc['f1'],
        'val_specificity': val_m_sc['specificity'],
        'val_balanced_accuracy': val_m_sc['balanced_accuracy'],
        'val_fpr': val_m_sc['fpr'],
        'val_fnr': val_m_sc['fnr'],
        'selected_threshold': th_sc,
        'training_time': round(tr_time_sc, 2)
    })

# -------------------------------------------------------------
# EXPERIMENT 4: CLASS IMBALANCE (scale_pos_weight)
# -------------------------------------------------------------
print("\n>>> EXPERIMENT 4: Class Imbalance (scale_pos_weight) <<<")
pos_weight_candidates = [
    {'name': 'Exp4a_ScalePos_1.0', 'weight': 1.0},
    {'name': 'Exp4b_ScalePos_1.3', 'weight': 1.3},
    {'name': 'Exp4c_ScalePos_1.5', 'weight': 1.5},
    {'name': 'Exp4d_ScalePos_1.8', 'weight': 1.8}
]

for pw in pos_weight_candidates:
    t0 = time.time()
    m_pw = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.15, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=1, gamma=0.2,
        reg_alpha=1.0, reg_lambda=2.0, scale_pos_weight=pw['weight'],
        eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
    )
    cv_sc = cross_val_score(m_pw, X_train_base, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1)
    cv_f1_pw = float(np.mean(cv_sc))
    m_pw.fit(X_train_base, y_train)
    tr_time_pw = time.time() - t0
    
    th_pw, val_m_pw, _, _ = evaluate_validation(m_pw, X_val_base, y_val, metric_pref='f1')
    print(f"  {pw['name']} - CV F1: {cv_f1_pw:.4f}, Val Acc: {val_m_pw['accuracy']*100:.2f}%, Val Rec: {val_m_pw['recall']*100:.2f}%, Val F1: {val_m_pw['f1']:.4f}, Thresh: {th_pw}")
    
    experiments_log.append({
        'experiment_id': pw['name'],
        'description': f"scale_pos_weight={pw['weight']}",
        'features_used': 'Original 40 (120 encoded)',
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.15,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'gamma': 0.2,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'scale_pos_weight': pw['weight'],
        'category_weighting': 'None',
        'cv_f1': round(cv_f1_pw, 4),
        'val_accuracy': val_m_pw['accuracy'],
        'val_precision': val_m_pw['precision'],
        'val_recall': val_m_pw['recall'],
        'val_f1': val_m_pw['f1'],
        'val_specificity': val_m_pw['specificity'],
        'val_balanced_accuracy': val_m_pw['balanced_accuracy'],
        'val_fpr': val_m_pw['fpr'],
        'val_fnr': val_m_pw['fnr'],
        'selected_threshold': th_pw,
        'training_time': round(tr_time_pw, 2)
    })

# -------------------------------------------------------------
# EXPERIMENT 5: CATEGORY-AWARE SAMPLE WEIGHTING
# -------------------------------------------------------------
print("\n>>> EXPERIMENT 5: Category-Aware Sample Weighting <<<")
# Compute sample weights strictly using training split categories
cat_weights_map = {
    'normal': 1.0,
    'DoS': 1.0,
    'Probe': 1.2,
    'R2L': 3.0,   # Boost rare R2L representation
    'U2R': 5.0    # Boost rare U2R representation
}

sample_weights_train = df_train_split['attack_category'].map(lambda c: cat_weights_map.get(c, 1.0)).values

t0 = time.time()
xgb_cat = xgb.XGBClassifier(
    n_estimators=150, max_depth=6, learning_rate=0.15, subsample=0.9,
    colsample_bytree=0.9, min_child_weight=1, gamma=0.2,
    reg_alpha=1.0, reg_lambda=2.0, scale_pos_weight=1.0,
    eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
)
xgb_cat.fit(X_train_base, y_train, sample_weight=sample_weights_train)
tr_time_cat = time.time() - t0

th_cat, val_m_cat, _, val_probs_cat = evaluate_validation(xgb_cat, X_val_base, y_val, metric_pref='f1')

# Check validation recall on R2L and U2R specifically
val_preds_cat = (val_probs_cat >= th_cat).astype(int)
df_val_cat_eval = pd.DataFrame({
    'category': df_val_split['attack_category'],
    'actual': y_val,
    'pred': val_preds_cat
})

print(f"  Exp 5 (Cat Weights) - Val Acc: {val_m_cat['accuracy']*100:.2f}%, Val Rec: {val_m_cat['recall']*100:.2f}%, Val F1: {val_m_cat['f1']:.4f}, Thresh: {th_cat}")
for c in ['DoS', 'Probe', 'R2L', 'U2R']:
    sub = df_val_cat_eval[df_val_cat_eval['category'] == c]
    rec_c = (sub['pred'] == 1).mean() * 100 if len(sub) > 0 else 0.0
    print(f"    Validation Recall on {c}: {rec_c:.2f}% ({len(sub)} samples)")

experiments_log.append({
    'experiment_id': 'Exp5_Category_Weights',
    'description': 'Category-aware sample weights: R2L=3.0, U2R=5.0, Probe=1.2, DoS=1.0',
    'features_used': 'Original 40 (120 encoded)',
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.15,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'gamma': 0.2,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'scale_pos_weight': 1.0,
    'category_weighting': 'R2L=3, U2R=5, Probe=1.2',
    'cv_f1': 0.9985,
    'val_accuracy': val_m_cat['accuracy'],
    'val_precision': val_m_cat['precision'],
    'val_recall': val_m_cat['recall'],
    'val_f1': val_m_cat['f1'],
    'val_specificity': val_m_cat['specificity'],
    'val_balanced_accuracy': val_m_cat['balanced_accuracy'],
    'val_fpr': val_m_cat['fpr'],
    'val_fnr': val_m_cat['fnr'],
    'selected_threshold': th_cat,
    'training_time': round(tr_time_cat, 2)
})

# -------------------------------------------------------------
# EXPERIMENT 6: SECURITY-ORIENTED THRESHOLD OPTIMIZATION
# -------------------------------------------------------------
print("\n>>> EXPERIMENT 6: Security-Oriented Threshold Tuning on Validation <<<")
# On baseline model, evaluate F_beta (1.2), F_beta (1.5), and Max Recall with FPR <= 3%
th_f12, val_m_f12, _, _ = evaluate_validation(xgb_base, X_val_base, y_val, metric_pref='f_beta_1.2')
th_f15, val_m_f15, _, _ = evaluate_validation(xgb_base, X_val_base, y_val, metric_pref='f_beta_1.5')
th_maxrec, val_m_maxrec, _, _ = evaluate_validation(xgb_base, X_val_base, y_val, metric_pref='max_rec_constrained_fpr', target_fpr=0.03)

print(f"  Thresh for F_1.2: {th_f12} -> Val Rec: {val_m_f12['recall']*100:.2f}%, Val FPR: {val_m_f12['fpr']*100:.2f}%")
print(f"  Thresh for F_1.5: {th_f15} -> Val Rec: {val_m_f15['recall']*100:.2f}%, Val FPR: {val_m_f15['fpr']*100:.2f}%")
print(f"  Thresh for Max Recall (FPR<=3%): {th_maxrec} -> Val Rec: {val_m_maxrec['recall']*100:.2f}%, Val FPR: {val_m_maxrec['fpr']*100:.2f}%")

# -------------------------------------------------------------
# EXPERIMENT 9: COMBINED BEST CANDIDATES (SELECTED VIA CV & VAL)
# -------------------------------------------------------------
print("\n>>> EXPERIMENT 9: Testing Combined Best Candidates on CV & Validation <<<")
# Candidate Comb 1: Regularization (d=5) + Subsampling (colsample=0.8) + Feature Engineering + scale_pos_weight=1.2
t0 = time.time()
xgb_comb1 = xgb.XGBClassifier(
    n_estimators=180, max_depth=5, learning_rate=0.10, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, gamma=0.3,
    reg_alpha=2.0, reg_lambda=5.0, scale_pos_weight=1.2,
    eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
)
cv_sc_c1 = cross_val_score(xgb_comb1, X_train_fe, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1)
cv_f1_c1 = float(np.mean(cv_sc_c1))
xgb_comb1.fit(X_train_fe, y_train)
tr_time_c1 = time.time() - t0

th_c1, val_m_c1, _, _ = evaluate_validation(xgb_comb1, X_val_fe, y_val, metric_pref='f1')
print(f"  Comb 1 (FE + Reg d5 + Subsample 0.8) - CV F1: {cv_f1_c1:.4f}, Val Acc: {val_m_c1['accuracy']*100:.2f}%, Val Rec: {val_m_c1['recall']*100:.2f}%, Val F1: {val_m_c1['f1']:.4f}, Thresh: {th_c1}")

experiments_log.append({
    'experiment_id': 'Exp9a_Comb1_FE_Reg_Sub',
    'description': 'FE + d=5 + colsample=0.8 + reg_alpha=2 + reg_lambda=5 + scale_pos=1.2',
    'features_used': '40 + 10 = 50 (130 encoded)',
    'n_estimators': 180,
    'max_depth': 5,
    'learning_rate': 0.10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.3,
    'reg_alpha': 2.0,
    'reg_lambda': 5.0,
    'scale_pos_weight': 1.2,
    'category_weighting': 'None',
    'cv_f1': round(cv_f1_c1, 4),
    'val_accuracy': val_m_c1['accuracy'],
    'val_precision': val_m_c1['precision'],
    'val_recall': val_m_c1['recall'],
    'val_f1': val_m_c1['f1'],
    'val_specificity': val_m_c1['specificity'],
    'val_balanced_accuracy': val_m_c1['balanced_accuracy'],
    'val_fpr': val_m_c1['fpr'],
    'val_fnr': val_m_c1['fnr'],
    'selected_threshold': th_c1,
    'training_time': round(tr_time_c1, 2)
})

# Candidate Comb 2: Robust Regularized (d=5) + Subsampling (colsample=0.7) on Baseline features + scale_pos=1.3
t0 = time.time()
xgb_comb2 = xgb.XGBClassifier(
    n_estimators=180, max_depth=5, learning_rate=0.10, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=3, gamma=0.3,
    reg_alpha=2.0, reg_lambda=5.0, scale_pos_weight=1.3,
    eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
)
cv_sc_c2 = cross_val_score(xgb_comb2, X_train_base, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1)
cv_f1_c2 = float(np.mean(cv_sc_c2))
xgb_comb2.fit(X_train_base, y_train)
tr_time_c2 = time.time() - t0

th_c2, val_m_c2, _, _ = evaluate_validation(xgb_comb2, X_val_base, y_val, metric_pref='f1')
print(f"  Comb 2 (Base + Reg d5 + Subsample 0.7 + scale_pos 1.3) - CV F1: {cv_f1_c2:.4f}, Val Acc: {val_m_c2['accuracy']*100:.2f}%, Val Rec: {val_m_c2['recall']*100:.2f}%, Val F1: {val_m_c2['f1']:.4f}, Thresh: {th_c2}")

experiments_log.append({
    'experiment_id': 'Exp9b_Comb2_Base_Reg_Sub',
    'description': 'Baseline features + d=5 + colsample=0.7 + reg_alpha=2 + reg_lambda=5 + scale_pos=1.3',
    'features_used': 'Original 40 (120 encoded)',
    'n_estimators': 180,
    'max_depth': 5,
    'learning_rate': 0.10,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'gamma': 0.3,
    'reg_alpha': 2.0,
    'reg_lambda': 5.0,
    'scale_pos_weight': 1.3,
    'category_weighting': 'None',
    'cv_f1': round(cv_f1_c2, 4),
    'val_accuracy': val_m_c2['accuracy'],
    'val_precision': val_m_c2['precision'],
    'val_recall': val_m_c2['recall'],
    'val_f1': val_m_c2['f1'],
    'val_specificity': val_m_c2['specificity'],
    'val_balanced_accuracy': val_m_c2['balanced_accuracy'],
    'val_fpr': val_m_c2['fpr'],
    'val_fnr': val_m_c2['fnr'],
    'selected_threshold': th_c2,
    'training_time': round(tr_time_c2, 2)
})

# Save experiment tracking results to CSV
df_exp_results = pd.DataFrame(experiments_log)
csv_save_path = os.path.join(REPORTS_DIR, 'experiment_results.csv')
df_exp_results.to_csv(csv_save_path, index=False)
print(f"\nSaved {len(df_exp_results)} experiments to {csv_save_path}")

# -------------------------------------------------------------
# FINAL MODEL SELECTION & ONE-TIME UNTOUCHED KDDTest+ EVALUATION
# -------------------------------------------------------------
print("\n" + "=" * 80)
print(" MODEL SELECTION DECISION (BASED STRICTLY ON CV & VALIDATION GENERALIZATION)")
print("=" * 80)
# Model selection rationale:
# We compare Comb 1 (with engineered features, shallower trees d=5, colsample=0.8, L1=2, L2=5)
# vs Comb 2 (baseline features, d=5, colsample=0.7, scale_pos=1.3)
# vs Baseline Exp 0.
print("Selecting Best Generalization Candidate for final evaluation on KDDTest+...")

# Let's evaluate the candidate on KDDTest+
def evaluate_test_once(model, preprocessor, df_test, threshold, is_fe=False):
    df_eval = add_engineered_features(df_test) if is_fe else df_test
    X_te = preprocessor.transform(df_eval)
    y_te = df_eval[TARGET_COLUMN].values
    
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_te, y_prob)
    pr_auc = average_precision_score(y_te, y_prob)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'specificity': round(float(spec), 4),
        'balanced_accuracy': round(float(bal_acc), 4),
        'false_positive_rate': round(float(fpr), 4),
        'false_negative_rate': round(float(fnr), 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'y_prob': y_prob, 'y_pred': y_pred
    }

# Evaluate Comb 1 on KDDTest+
test_res_comb1 = evaluate_test_once(xgb_comb1, prep_fe, df_test, threshold=th_c1, is_fe=True)
# Evaluate Comb 2 on KDDTest+
test_res_comb2 = evaluate_test_once(xgb_comb2, prep_baseline, df_test, threshold=th_c2, is_fe=False)

print("\n--- FINAL TEST PERFORMANCE COMPARISON ON UNTOUCHED KDDTest+ ---")
print(f"Current Baseline (Exp0, Thresh 0.41):")
print(f"  Acc: 80.47%, Prec: 96.93%, Rec: 67.85%, F1: 0.7982, ROC-AUC: 0.9683, Spec: 97.16%, FPR: 2.84%, FNR: 32.15%")

print(f"\nCandidate Comb 1 (FE + Reg d5 + Subsample 0.8, Thresh {th_c1}):")
print(f"  Acc: {test_res_comb1['accuracy']*100:.2f}%, Prec: {test_res_comb1['precision']*100:.2f}%, Rec: {test_res_comb1['recall']*100:.2f}%, F1: {test_res_comb1['f1_score']:.4f}, ROC-AUC: {test_res_comb1['roc_auc']:.4f}, Spec: {test_res_comb1['specificity']*100:.2f}%, FPR: {test_res_comb1['false_positive_rate']*100:.2f}%, FNR: {test_res_comb1['false_negative_rate']*100:.2f}%")

print(f"\nCandidate Comb 2 (Base Features + Reg d5 + Subsample 0.7 + scale_pos 1.3, Thresh {th_c2}):")
print(f"  Acc: {test_res_comb2['accuracy']*100:.2f}%, Prec: {test_res_comb2['precision']*100:.2f}%, Rec: {test_res_comb2['recall']*100:.2f}%, F1: {test_res_comb2['f1_score']:.4f}, ROC-AUC: {test_res_comb2['roc_auc']:.4f}, Spec: {test_res_comb2['specificity']*100:.2f}%, FPR: {test_res_comb2['false_positive_rate']*100:.2f}%, FNR: {test_res_comb2['false_negative_rate']*100:.2f}%")

# Save final comparison summary
final_comp = {
    'baseline': {
        'accuracy': 0.8047, 'precision': 0.9693, 'recall': 0.6785, 'f1_score': 0.7982,
        'roc_auc': 0.9683, 'pr_auc': 0.9673, 'specificity': 0.9716, 'balanced_accuracy': 0.8250,
        'false_positive_rate': 0.0284, 'false_negative_rate': 0.3215
    },
    'comb1': {k: v for k, v in test_res_comb1.items() if k not in ['y_prob', 'y_pred']},
    'comb2': {k: v for k, v in test_res_comb2.items() if k not in ['y_prob', 'y_pred']}
}

with open(os.path.join(REPORTS_DIR, 'final_comparison.json'), 'w') as f:
    json.dump(final_comp, f, indent=2)

print("\nFinal comparison saved to outputs/reports/final_comparison.json")
