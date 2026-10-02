import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RAW_COLUMNS,
    LABEL_COLUMN, TARGET_COLUMN, ATTACK_CATEGORIES,
    RANDOM_STATE, REPORTS_DIR, ALL_FEATURES
)

def load_raw_data(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH):
    """
    Loads KDDTrain+ and KDDTest+ files with column definitions.
    Adds binary_target (0=Normal, 1=Attack) and attack_category.
    """
    print(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path, names=RAW_COLUMNS, header=None)
    
    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path, names=RAW_COLUMNS, header=None)
    
    for df in [df_train, df_test]:
        # Binary target: 0 for normal, 1 for any attack
        df[TARGET_COLUMN] = (df[LABEL_COLUMN] != 'normal').astype(int)
        # High-level attack category mapping
        df['attack_category'] = df[LABEL_COLUMN].map(lambda x: ATTACK_CATEGORIES.get(x, 'Unknown'))
        
    print(f"Loaded Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test

def get_train_val_split(df_train, val_size=0.20, random_state=RANDOM_STATE):
    """
    Splits the training data into an internal Train set and a Validation set.
    Uses stratification on the binary target to maintain exact class proportions.
    The final test set (KDDTest+) remains untouched.
    """
    print(f"Splitting training data into Train ({int((1-val_size)*100)}%) and Validation ({int(val_size*100)}%)...")
    df_train_split, df_val_split = train_test_split(
        df_train,
        test_size=val_size,
        random_state=random_state,
        stratify=df_train[TARGET_COLUMN]
    )
    print(f"Train split shape: {df_train_split.shape}, Validation split shape: {df_val_split.shape}")
    return df_train_split.reset_index(drop=True), df_val_split.reset_index(drop=True)

def extract_real_presets(df_test, save_path=None):
    """
    Extracts real connection records directly from KDDTest+ for interactive demonstration.
    Picks:
      1. Normal HTTP Traffic
      2. DoS / Neptune Attack
      3. Probe Attack (e.g. portsweep or satan)
      4. R2L Attack (e.g. guess_passwd or warezclient)
      5. U2R Attack (e.g. buffer_overflow or rootkit)
      6. Random Test Record
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, 'presets.json')
        
    presets = {}
    
    # 1. Normal HTTP
    normal_candidates = df_test[(df_test[LABEL_COLUMN] == 'normal') & (df_test['service'] == 'http')]
    if not normal_candidates.empty:
        presets['Normal HTTP Traffic'] = normal_candidates.iloc[0].to_dict()
    else:
        presets['Normal HTTP Traffic'] = df_test[df_test[LABEL_COLUMN] == 'normal'].iloc[0].to_dict()
        
    # 2. DoS / Neptune
    neptune_candidates = df_test[df_test[LABEL_COLUMN] == 'neptune']
    if not neptune_candidates.empty:
        presets['DoS / Neptune Attack'] = neptune_candidates.iloc[0].to_dict()
    else:
        presets['DoS / Neptune Attack'] = df_test[df_test['attack_category'] == 'DoS'].iloc[0].to_dict()
        
    # 3. Probe (e.g. portsweep or satan or saint)
    probe_candidates = df_test[df_test['attack_category'] == 'Probe']
    if not probe_candidates.empty:
        presets['Probe / Portscan Attack'] = probe_candidates.iloc[0].to_dict()
        
    # 4. R2L (e.g. guess_passwd)
    r2l_candidates = df_test[df_test[LABEL_COLUMN] == 'guess_passwd']
    if not r2l_candidates.empty:
        presets['R2L / Password Guessing Attack'] = r2l_candidates.iloc[0].to_dict()
    else:
        presets['R2L / Password Guessing Attack'] = df_test[df_test['attack_category'] == 'R2L'].iloc[0].to_dict()
        
    # 5. U2R (e.g. buffer_overflow)
    u2r_candidates = df_test[df_test[LABEL_COLUMN] == 'buffer_overflow']
    if not u2r_candidates.empty:
        presets['U2R / Buffer Overflow Attack'] = u2r_candidates.iloc[0].to_dict()
    else:
        presets['U2R / Buffer Overflow Attack'] = df_test[df_test['attack_category'] == 'U2R'].iloc[0].to_dict()
        
    # 6. Random Test Record
    random_row = df_test.sample(n=1, random_state=RANDOM_STATE).iloc[0].to_dict()
    presets['Random Test Sample'] = random_row

    # Clean serializable dictionary
    cleaned_presets = {}
    for name, record in presets.items():
        # Ensure all types are native python types for JSON serialization
        cleaned = {}
        for k, v in record.items():
            if isinstance(v, (np.integer, int)):
                cleaned[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                cleaned[k] = float(v)
            else:
                cleaned[k] = str(v)
        cleaned_presets[name] = cleaned

    with open(save_path, 'w') as f:
        json.dump(cleaned_presets, f, indent=2)
    print(f"Extracted and saved {len(cleaned_presets)} real KDDTest+ presets to {save_path}")
    return cleaned_presets
