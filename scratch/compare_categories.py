import os
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = r"c:\Users\dashw\OneDrive\Desktop\Dashwanth\College\sem 5\cryptography\intrstrusion detection using xgboost"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RAW_COLUMNS,
    LABEL_COLUMN, TARGET_COLUMN, ATTACK_CATEGORIES, RANDOM_STATE
)
from src.preprocessor import IDSPreprocessor

df_train = pd.read_csv(TRAIN_DATA_PATH, names=RAW_COLUMNS, header=None)
df_train[TARGET_COLUMN] = (df_train[LABEL_COLUMN] != 'normal').astype(int)
df_train['attack_category'] = df_train[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))

df_test = pd.read_csv(TEST_DATA_PATH, names=RAW_COLUMNS, header=None)
df_test[TARGET_COLUMN] = (df_test[LABEL_COLUMN] != 'normal').astype(int)
df_test['attack_category'] = df_test[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))

train_attacks = set(df_train[LABEL_COLUMN].unique())
test_attacks = set(df_test[LABEL_COLUMN].unique())
unseen_attacks = test_attacks - train_attacks

prep = IDSPreprocessor.load()
X_te = prep.transform(df_test)
y_te = df_test[TARGET_COLUMN].values

# 1. Baseline Model
base_m = xgb.XGBClassifier()
base_m.load_model(os.path.join(PROJECT_ROOT, 'outputs', 'models', 'tuned_xgb.json'))
prob_base = base_m.predict_proba(X_te)[:, 1]
pred_base = (prob_base >= 0.41).astype(int)

# 2. Exp2a Model
from sklearn.model_selection import train_test_split
df_train_split, df_val_split = train_test_split(
    df_train, test_size=0.20, random_state=RANDOM_STATE, stratify=df_train[TARGET_COLUMN]
)
X_tr = prep.transform(df_train_split)
y_tr = df_train_split[TARGET_COLUMN].values

m_exp2a = xgb.XGBClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.15, subsample=0.9,
    colsample_bytree=0.9, min_child_weight=3, gamma=0.5,
    reg_alpha=3.0, reg_lambda=5.0, scale_pos_weight=1.15,
    eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
)
m_exp2a.fit(X_tr, y_tr)
prob_2a = m_exp2a.predict_proba(X_te)[:, 1]
pred_2a = (prob_2a >= 0.36).astype(int)

# Compare category-by-category
df_comp = pd.DataFrame({
    'attack_type': df_test[LABEL_COLUMN],
    'attack_category': df_test['attack_category'],
    'is_attack': y_te,
    'is_unseen': df_test[LABEL_COLUMN].isin(unseen_attacks),
    'pred_base': pred_base,
    'pred_2a': pred_2a
})

print("=" * 80)
print(" CATEGORY-BY-CATEGORY BREAKDOWN: BASELINE vs EXP2A (SHALLOWER REGULARIZED)")
print("=" * 80)

for cat in ['normal', 'DoS', 'Probe', 'R2L', 'U2R']:
    sub = df_comp[df_comp['attack_category'] == cat]
    total = len(sub)
    if cat == 'normal':
        b_acc = (sub['pred_base'] == 0).sum()
        a_acc = (sub['pred_2a'] == 0).sum()
        print(f"NORMAL (total {total}):")
        print(f"  Baseline: {b_acc} correct ({b_acc/total*100:.2f}%), {total - b_acc} false alarms")
        print(f"  Exp 2a:   {a_acc} correct ({a_acc/total*100:.2f}%), {total - a_acc} false alarms")
    else:
        b_det = (sub['pred_base'] == 1).sum()
        a_det = (sub['pred_2a'] == 1).sum()
        print(f"{cat:<6} (total {total}):")
        print(f"  Baseline: {b_det} detected ({b_det/total*100:.2f}%), {total - b_det} missed")
        print(f"  Exp 2a:   {a_det} detected ({a_det/total*100:.2f}%), {total - a_det} missed (Diff: {a_det - b_det:+d})")

# Known vs Unseen
known_sub = df_comp[(df_comp['is_attack'] == 1) & (~df_comp['is_unseen'])]
unseen_sub = df_comp[(df_comp['is_attack'] == 1) & (df_comp['is_unseen'])]

print("\nKNOWN ATTACKS (total 9,083):")
print(f"  Baseline: {(known_sub['pred_base'] == 1).sum()} ({(known_sub['pred_base'] == 1).mean()*100:.2f}%)")
print(f"  Exp 2a:   {(known_sub['pred_2a'] == 1).sum()} ({(known_sub['pred_2a'] == 1).mean()*100:.2f}%) (Diff: {(known_sub['pred_2a'] == 1).sum() - (known_sub['pred_base'] == 1).sum():+d})")

print("\nUNSEEN / NOVEL ATTACKS (total 3,750):")
print(f"  Baseline: {(unseen_sub['pred_base'] == 1).sum()} ({(unseen_sub['pred_base'] == 1).mean()*100:.2f}%)")
print(f"  Exp 2a:   {(unseen_sub['pred_2a'] == 1).sum()} ({(unseen_sub['pred_2a'] == 1).mean()*100:.2f}%) (Diff: {(unseen_sub['pred_2a'] == 1).sum() - (unseen_sub['pred_base'] == 1).sum():+d})")
