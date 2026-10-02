import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

PROJECT_ROOT = r"c:\Users\dashw\OneDrive\Desktop\Dashwanth\College\sem 5\cryptography\intrstrusion detection using xgboost"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RAW_COLUMNS,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ALL_FEATURES,
    LABEL_COLUMN, TARGET_COLUMN, ATTACK_CATEGORIES, RANDOM_STATE
)
from src.preprocessor import IDSPreprocessor

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

prep = IDSPreprocessor()
prep.fit(df_train_split)
X_tr = prep.transform(df_train_split)
y_tr = df_train_split[TARGET_COLUMN].values
X_va = prep.transform(df_val_split)
y_va = df_val_split[TARGET_COLUMN].values
X_te = prep.transform(df_test)
y_te = df_test[TARGET_COLUMN].values

# Test Refined Candidate:
# Combining Shallower Trees (d=4, alpha=3, lambda=5, min_child=3, gamma=0.5)
# with Feature Subsampling (colsample_bytree=0.7, subsample=0.85)
m_refined = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.15,
    subsample=0.85,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.5,
    reg_alpha=3.0,
    reg_lambda=5.0,
    scale_pos_weight=1.15,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_f1 = np.mean(cross_val_score(m_refined, X_tr, y_tr, cv=cv_strat, scoring='f1', n_jobs=-1))
m_refined.fit(X_tr, y_tr)

# Validation threshold optimization
val_probs = m_refined.predict_proba(X_va)[:, 1]
best_f1, best_th = -1, 0.5
for t in np.linspace(0.05, 0.95, 91):
    f = f1_score(y_va, (val_probs >= t).astype(int), zero_division=0)
    if f > best_f1:
        best_f1, best_th = f, round(float(t), 2)

# Test evaluation
te_probs = m_refined.predict_proba(X_te)[:, 1]
te_preds = (te_probs >= best_th).astype(int)

tn, fp, fn, tp = confusion_matrix(y_te, te_preds, labels=[0, 1]).ravel()
acc = accuracy_score(y_te, te_preds)
prec = precision_score(y_te, te_preds, zero_division=0)
rec = recall_score(y_te, te_preds, zero_division=0)
f1 = f1_score(y_te, te_preds, zero_division=0)
roc_auc = roc_auc_score(y_te, te_probs)
pr_auc = average_precision_score(y_te, te_probs)
spec = tn / (tn + fp)
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"Refined Candidate (d=4 + colsample=0.7 + alpha=3 + lambda=5):")
print(f"  CV F1: {cv_f1:.4f} | Val Thresh: {best_th} | Val F1: {best_f1:.4f}")
print(f"  Test Accuracy: {acc*100:.2f}% (Baseline: 80.47%)")
print(f"  Test Precision: {prec*100:.2f}% (Baseline: 96.93%)")
print(f"  Test Recall: {rec*100:.2f}% (Baseline: 67.85%)")
print(f"  Test F1: {f1:.4f} (Baseline: 0.7982)")
print(f"  Test ROC-AUC: {roc_auc:.4f} (Baseline: 0.9683)")
print(f"  Test Specificity: {spec*100:.2f}% (Baseline: 97.16%)")
print(f"  Test FPR: {fpr*100:.2f}% (Baseline: 2.84%)")
print(f"  Test FNR: {fnr*100:.2f}% (Baseline: 32.15%)")
print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
