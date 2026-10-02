import os
import json
import pandas as pd
import numpy as np
from src.config import (
    REPORTS_DIR, LABEL_COLUMN, TARGET_COLUMN,
    ATTACK_CATEGORIES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
)

def run_eda(df_train, df_test):
    """
    Performs comprehensive Exploratory Data Analysis on the NSL-KDD dataset.
    Returns and saves structured statistical summary.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Running Exploratory Data Analysis (EDA)...")
    
    eda_summary = {
        'dataset_dimensions': {
            'train_rows': len(df_train),
            'train_cols': df_train.shape[1],
            'test_rows': len(df_test),
            'test_cols': df_test.shape[1]
        },
        'missing_values': {
            'train_null_count': int(df_train.isnull().sum().sum()),
            'test_null_count': int(df_test.isnull().sum().sum())
        },
        'duplicates': {
            'train_duplicates': int(df_train.duplicated().sum()),
            'test_duplicates': int(df_test.duplicated().sum())
        },
        'binary_class_distribution': {
            'train': {
                'normal': int((df_train[TARGET_COLUMN] == 0).sum()),
                'attack': int((df_train[TARGET_COLUMN] == 1).sum()),
                'normal_pct': round(float((df_train[TARGET_COLUMN] == 0).mean()) * 100, 2),
                'attack_pct': round(float((df_train[TARGET_COLUMN] == 1).mean()) * 100, 2)
            },
            'test': {
                'normal': int((df_test[TARGET_COLUMN] == 0).sum()),
                'attack': int((df_test[TARGET_COLUMN] == 1).sum()),
                'normal_pct': round(float((df_test[TARGET_COLUMN] == 0).mean()) * 100, 2),
                'attack_pct': round(float((df_test[TARGET_COLUMN] == 1).mean()) * 100, 2)
            }
        },
        'attack_category_distribution': {
            'train': df_train['attack_category'].value_counts().to_dict(),
            'test': df_test['attack_category'].value_counts().to_dict()
        },
        'categorical_cardinality': {
            col: {
                'train_unique_count': int(df_train[col].nunique()),
                'test_unique_count': int(df_test[col].nunique()),
                'train_categories': sorted(list(df_train[col].unique())),
                'unseen_in_train': sorted(list(set(df_test[col].unique()) - set(df_train[col].unique())))
            }
            for col in CATEGORICAL_FEATURES
        },
        'novel_attack_types_in_test': sorted(list(set(df_test[LABEL_COLUMN].unique()) - set(df_train[LABEL_COLUMN].unique()))),
        'numerical_feature_stats': {}
    }

    # Compute key stats for top numerical features
    for col in NUMERICAL_FEATURES:
        eda_summary['numerical_feature_stats'][col] = {
            'mean': float(df_train[col].mean()),
            'std': float(df_train[col].std()),
            'min': float(df_train[col].min()),
            'median': float(df_train[col].median()),
            'max': float(df_train[col].max()),
            'skewness': float(df_train[col].skew())
        }

    # Save summary to reports
    summary_path = os.path.join(REPORTS_DIR, 'eda_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(eda_summary, f, indent=2)
    print(f"EDA summary successfully saved to {summary_path}")

    return eda_summary
