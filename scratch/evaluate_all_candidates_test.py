import os
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb
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
    LABEL_COLUMN, TARGET_COLUMN, ATTACK_CATEGORIES, RANDOM_STATE, REPORTS_DIR
)
from src.preprocessor import IDSPreprocessor

print("=" * 80)
print(" EVALUATING ALL CANDIDATE EXPERIMENTS ON UNTOUCHED KDDTest+")
print("=" * 80)

# Load data
df_train = pd.read_csv(TRAIN_DATA_PATH, names=RAW_COLUMNS, header=None)
df_train[TARGET_COLUMN] = (df_train[LABEL_COLUMN] != 'normal').astype(int)
df_train['attack_category'] = df_train[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))

df_test = pd.read_csv(TEST_DATA_PATH, names=RAW_COLUMNS, header=None)
df_test[TARGET_COLUMN] = (df_test[LABEL_COLUMN] != 'normal').astype(int)
df_test['attack_category'] = df_test[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))

df_train_split, df_val_split = train_test_split(
    df_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df_train[TARGET_COLUMN]
)

prep_baseline = IDSPreprocessor()
prep_baseline.fit(df_train_split)
X_train_base = prep_baseline.transform(df_train_split)
y_train = df_train_split[TARGET_COLUMN].values
X_val_base = prep_baseline.transform(df_val_split)
y_val = df_val_split[TARGET_COLUMN].values
X_test_base = prep_baseline.transform(df_test)
y_test = df_test[TARGET_COLUMN].values

# Feature engineering helper
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
df_test_fe = add_engineered_features(df_test)

prep_fe = IDSPreprocessor()
prep_fe.numerical_features = NUMERICAL_FEATURES + [
    'bytes_total', 'bytes_ratio', 'src_dst_diff', 'count_srv_ratio',
    'same_srv_count', 'diff_srv_count', 'dst_host_same_srv_count',
    'dst_host_diff_srv_count', 'syn_err_count', 'rej_err_count'
]
prep_fe.fit(df_train_fe)
X_train_fe = prep_fe.transform(df_train_fe)
X_val_fe = prep_fe.transform(df_val_fe)
X_test_fe = prep_fe.transform(df_test_fe)

# Read experiment results CSV to get parameters and selected thresholds
df_exp = pd.read_csv(os.path.join(REPORTS_DIR, 'experiment_results.csv'))

results = []

for idx, row in df_exp.iterrows():
    exp_id = row['experiment_id']
    th = row['selected_threshold']
    is_fe = 'FE' in exp_id or 'Feature_Engineering' in exp_id
    
    X_tr = X_train_fe if is_fe else X_train_base
    X_va = X_val_fe if is_fe else X_val_base
    X_te = X_test_fe if is_fe else X_test_base
    
    m = xgb.XGBClassifier(
        n_estimators=int(row['n_estimators']),
        max_depth=int(row['max_depth']),
        learning_rate=float(row['learning_rate']),
        subsample=float(row['subsample']),
        colsample_bytree=float(row['colsample_bytree']),
        min_child_weight=float(row['min_child_weight']),
        gamma=float(row['gamma']),
        reg_alpha=float(row['reg_alpha']),
        reg_lambda=float(row['reg_lambda']),
        scale_pos_weight=float(row['scale_pos_weight']),
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Category weighting if applicable
    if 'Category' in exp_id:
        cat_map = {'normal': 1.0, 'DoS': 1.0, 'Probe': 1.2, 'R2L': 3.0, 'U2R': 5.0}
        sw = df_train_split['attack_category'].map(lambda c: cat_map.get(c, 1.0)).values
        m.fit(X_tr, y_train, sample_weight=sw)
    else:
        m.fit(X_tr, y_train)
        
    y_prob = m.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= th).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    res_dict = {
        'experiment_id': exp_id,
        'description': row['description'],
        'threshold': th,
        'test_accuracy': round(float(acc) * 100, 2),
        'test_precision': round(float(prec) * 100, 2),
        'test_recall': round(float(rec) * 100, 2),
        'test_f1': round(float(f1), 4),
        'test_roc_auc': round(float(roc_auc), 4),
        'test_specificity': round(float(spec) * 100, 2),
        'test_fpr': round(float(fpr) * 100, 2),
        'test_fnr': round(float(fnr) * 100, 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    results.append(res_dict)
    print(f"{exp_id:<25} | Acc: {res_dict['test_accuracy']:>5.2f}% | Prec: {res_dict['test_precision']:>5.2f}% | Rec: {res_dict['test_recall']:>5.2f}% | F1: {res_dict['test_f1']:.4f} | Thresh: {th}")

df_test_comp = pd.DataFrame(results)
comp_csv_path = os.path.join(REPORTS_DIR, 'all_experiments_test_comparison.csv')
df_test_comp.to_csv(comp_csv_path, index=False)
print(f"\nSaved all test evaluations to {comp_csv_path}")
