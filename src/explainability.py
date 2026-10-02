import os
import json
import numpy as np
import pandas as pd
import shap
from src.config import REPORTS_DIR, FEATURE_METADATA

def get_feature_importances(model, feature_names, top_n=20):
    """
    Extracts XGBoost feature importances by Gain and Weight.
    """
    booster = model.get_booster()
    
    # Gain: Average improvement in accuracy brought by a feature to the branches it is on
    gain_dict = booster.get_score(importance_type='gain')
    # Weight: Number of times a feature appears in a tree
    weight_dict = booster.get_score(importance_type='weight')
    
    # Map feature names (e.g. 'f0', 'f1' or actual names)
    name_map = {}
    for i, name in enumerate(feature_names):
        name_map[f'f{i}'] = name
        name_map[name] = name

    parsed_gain = []
    for f_code, val in gain_dict.items():
        fname = name_map.get(f_code, f_code)
        parsed_gain.append({'feature': fname, 'importance': float(val)})
    df_gain = pd.DataFrame(parsed_gain).sort_values(by='importance', ascending=False).head(top_n)

    parsed_weight = []
    for f_code, val in weight_dict.items():
        fname = name_map.get(f_code, f_code)
        parsed_weight.append({'feature': fname, 'importance': float(val)})
    df_weight = pd.DataFrame(parsed_weight).sort_values(by='importance', ascending=False).head(top_n)

    results = {
        'top_gain': df_gain.to_dict(orient='records'),
        'top_weight': df_weight.to_dict(orient='records')
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results

def compute_shap_global(model, X_sample, feature_names):
    """
    Computes global SHAP values using TreeExplainer on a sample of test data.
    """
    print("Computing SHAP values for global explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Mean absolute SHAP values per feature
    if hasattr(shap_values, 'values'):
        vals = shap_values.values
        if vals.ndim == 3:  # binary classification with 2 outputs
            vals = vals[:, :, 1]
    else:
        vals = shap_values

    mean_abs_shap = np.abs(vals).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]
    
    top_shap_features = [
        {'feature': feature_names[i], 'mean_abs_shap': float(mean_abs_shap[i])}
        for i in top_indices
    ]
    
    with open(os.path.join(REPORTS_DIR, 'shap_summary.json'), 'w') as f:
        json.dump(top_shap_features, f, indent=2)
        
    return explainer, shap_values, top_shap_features

def explain_single_instance(explainer, model, preprocessor, input_record, top_k=5):
    """
    Explains a single network traffic record using SHAP values.
    Returns prediction, probability, top attack contributors, and top normal contributors.
    """
    X_trans = preprocessor.transform(input_record)
    feature_names = preprocessor.get_feature_names_out()
    
    prob = float(model.predict_proba(X_trans)[0, 1])
    
    shap_res = explainer(X_trans)
    if hasattr(shap_res, 'values'):
        vals = shap_res.values[0]
        if vals.ndim == 2:
            vals = vals[:, 1]
    else:
        vals = shap_res[0]

    # Positive SHAP pushes toward ATTACK (1)
    # Negative SHAP pushes toward NORMAL (0)
    pos_indices = np.argsort(vals)[::-1]
    neg_indices = np.argsort(vals)
    
    attack_contributors = []
    for idx in pos_indices:
        if vals[idx] <= 0:
            break
        fname = feature_names[idx]
        val_in_input = X_trans[0, idx]
        attack_contributors.append({
            'feature': fname,
            'shap_value': round(float(vals[idx]), 4),
            'feature_value': round(float(val_in_input), 4) if isinstance(val_in_input, (int, float, np.floating)) else str(val_in_input),
            'direction': 'Toward Attack',
            'explanation': f"Feature '{fname}' ({val_in_input}) contributed positively (+{vals[idx]:.3f}) toward the model's attack prediction."
        })
        if len(attack_contributors) >= top_k:
            break

    normal_contributors = []
    for idx in neg_indices:
        if vals[idx] >= 0:
            break
        fname = feature_names[idx]
        val_in_input = X_trans[0, idx]
        normal_contributors.append({
            'feature': fname,
            'shap_value': round(float(vals[idx]), 4),
            'feature_value': round(float(val_in_input), 4) if isinstance(val_in_input, (int, float, np.floating)) else str(val_in_input),
            'direction': 'Toward Normal',
            'explanation': f"Feature '{fname}' ({val_in_input}) contributed negatively ({vals[idx]:.3f}) toward the model's normal prediction."
        })
        if len(normal_contributors) >= top_k:
            break

    return {
        'attack_probability': prob,
        'attack_contributors': attack_contributors,
        'normal_contributors': normal_contributors
    }
