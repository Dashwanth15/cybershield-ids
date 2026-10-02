import os
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb

PROJECT_ROOT = r"c:\Users\dashw\OneDrive\Desktop\Dashwanth\College\sem 5\cryptography\intrstrusion detection using xgboost"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RAW_COLUMNS,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ALL_FEATURES,
    LABEL_COLUMN, TARGET_COLUMN, ATTACK_CATEGORIES, RANDOM_STATE
)
from sklearn.model_selection import train_test_split

print("=" * 80)
print("INVESTIGATING THE GENERALIZATION GAP IN NSL-KDD")
print("=" * 80)

# Load data
df_train = pd.read_csv(TRAIN_DATA_PATH, names=RAW_COLUMNS, header=None)
df_test = pd.read_csv(TEST_DATA_PATH, names=RAW_COLUMNS, header=None)

df_train[TARGET_COLUMN] = (df_train[LABEL_COLUMN] != 'normal').astype(int)
df_test[TARGET_COLUMN] = (df_test[LABEL_COLUMN] != 'normal').astype(int)

df_train['attack_category'] = df_train[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))
df_test['attack_category'] = df_test[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))

df_train_split, df_val_split = train_test_split(
    df_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df_train[TARGET_COLUMN]
)

print(f"Train split: {len(df_train_split)}, Val split: {len(df_val_split)}, Test: {len(df_test)}")

# 1. Target Class distribution
print("\n--- 1. Target Class Distribution ---")
print("Train Split:", df_train_split[TARGET_COLUMN].value_counts(normalize=True).to_dict())
print("Val Split:  ", df_val_split[TARGET_COLUMN].value_counts(normalize=True).to_dict())
print("Test:       ", df_test[TARGET_COLUMN].value_counts(normalize=True).to_dict())

# 2. Attack Category distribution
print("\n--- 2. Attack Category Breakdown (Counts & %) ---")
for split_name, df in [("Train Split", df_train_split), ("Val Split", df_val_split), ("Test", df_test)]:
    counts = df['attack_category'].value_counts()
    pcts = df['attack_category'].value_counts(normalize=True) * 100
    print(f"\n{split_name}:")
    for cat in ['normal', 'DoS', 'Probe', 'R2L', 'U2R']:
        c = counts.get(cat, 0)
        p = pcts.get(cat, 0.0)
        print(f"  {cat:<8}: {c:>6} ({p:>5.2f}%)")

# 3. Categorical features: Unseen categories
print("\n--- 3. Categorical Distributions & Unseen Categories ---")
for cat_col in CATEGORICAL_FEATURES:
    train_cats = set(df_train_split[cat_col].unique())
    val_cats = set(df_val_split[cat_col].unique())
    test_cats = set(df_test[cat_col].unique())
    
    unseen_in_val = val_cats - train_cats
    unseen_in_test = test_cats - train_cats
    train_not_in_test = train_cats - test_cats
    
    print(f"\nFeature: {cat_col}")
    print(f"  Train unique: {len(train_cats)}, Val unique: {len(val_cats)}, Test unique: {len(test_cats)}")
    print(f"  Unseen in Val: {unseen_in_val}")
    print(f"  Unseen in Test (not in train): {unseen_in_test}")
    print(f"  In Train but absent in Test: {len(train_not_in_test)} categories")
    if unseen_in_test:
        test_unseen_rows = df_test[df_test[cat_col].isin(unseen_in_test)]
        print(f"  Rows in Test with unseen {cat_col}: {len(test_unseen_rows)} ({len(test_unseen_rows)/len(df_test)*100:.2f}%)")

# 4. Attack types: Known vs Unseen in Test
print("\n--- 4. Attack Types in Test ---")
train_attacks = set(df_train_split[LABEL_COLUMN].unique())
test_attacks = set(df_test[LABEL_COLUMN].unique())
unseen_attacks = test_attacks - train_attacks
known_attacks = test_attacks & train_attacks

print(f"Total attack types in Train: {len(train_attacks)}")
print(f"Total attack types in Test: {len(test_attacks)}")
print(f"Novel / Unseen attack types in Test ({len(unseen_attacks)}): {sorted(list(unseen_attacks))}")

# Quantify test samples that are known vs unseen
df_test_attacks = df_test[df_test[LABEL_COLUMN] != 'normal']
test_known_attack_rows = df_test_attacks[df_test_attacks[LABEL_COLUMN].isin(known_attacks)]
test_unseen_attack_rows = df_test_attacks[df_test_attacks[LABEL_COLUMN].isin(unseen_attacks)]

print(f"\nTest Attack Samples Breakdown:")
print(f"  Total Attack Samples in Test: {len(df_test_attacks)}")
print(f"  Known Attack Samples in Test: {len(test_known_attack_rows)} ({len(test_known_attack_rows)/len(df_test_attacks)*100:.2f}%)")
print(f"  Novel/Unseen Attack Samples in Test: {len(test_unseen_attack_rows)} ({len(test_unseen_attack_rows)/len(df_test_attacks)*100:.2f}%)")

# Breakdown of unseen attacks by category
print("\nNovel Attacks Count by Category:")
print(test_unseen_attack_rows['attack_category'].value_counts())

# 5. Evaluate current model's performance on Known vs Unseen attacks
from src.preprocessor import IDSPreprocessor
preprocessor = IDSPreprocessor.load()
tuned_model = xgb.XGBClassifier()
tuned_model.load_model(os.path.join(PROJECT_ROOT, 'outputs', 'models', 'tuned_xgb.json'))

X_test = preprocessor.transform(df_test)
y_test_probs = tuned_model.predict_proba(X_test)[:, 1]
y_test_preds = (y_test_probs >= 0.41).astype(int)

df_eval = pd.DataFrame({
    'attack_type': df_test[LABEL_COLUMN],
    'attack_category': df_test['attack_category'],
    'is_attack': df_test[TARGET_COLUMN],
    'is_unseen': df_test[LABEL_COLUMN].isin(unseen_attacks),
    'pred_attack': y_test_preds,
    'prob': y_test_probs
})

# Performance on Normal
normal_sub = df_eval[df_eval['is_attack'] == 0]
print(f"\nNormal Traffic Accuracy: {(normal_sub['pred_attack'] == 0).mean()*100:.2f}% ({len(normal_sub)} samples)")

# Performance on Known Attacks
known_atk_sub = df_eval[(df_eval['is_attack'] == 1) & (~df_eval['is_unseen'])]
print(f"Known Attacks Detection Rate: {(known_atk_sub['pred_attack'] == 1).mean()*100:.2f}% ({len(known_atk_sub)} samples)")

# Performance on Unseen / Novel Attacks
unseen_atk_sub = df_eval[(df_eval['is_attack'] == 1) & (df_eval['is_unseen'])]
print(f"Unseen Attacks Detection Rate: {(unseen_atk_sub['pred_attack'] == 1).mean()*100:.2f}% ({len(unseen_atk_sub)} samples)")

# 6. Feature distribution shifts
print("\n--- 6. Feature Distribution Comparisons (Train vs Test) ---")
key_num = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'serror_rate', 'same_srv_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_serror_rate']
shift_data = []
for f in key_num:
    tr_m, tr_std = df_train_split[f].mean(), df_train_split[f].std()
    te_m, te_std = df_test[f].mean(), df_test[f].std()
    shift_data.append({
        'feature': f,
        'train_mean': round(float(tr_m), 2),
        'train_std': round(float(tr_std), 2),
        'test_mean': round(float(te_m), 2),
        'test_std': round(float(te_std), 2),
    })
print(pd.DataFrame(shift_data).to_string())
