import os
import sys
import io
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TEST_DATA_PATH,
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    FEATURE_METADATA, RAW_COLUMNS, LABEL_COLUMN, TARGET_COLUMN
)
from src.preprocessor import IDSPreprocessor
from src.explainability import explain_single_instance

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Global variables for loaded model artifacts
preprocessor = None
model = None
explainer = None
optimal_threshold = 0.50
master_results = {}
presets_data = {}

def load_artifacts():
    global preprocessor, model, explainer, optimal_threshold, master_results, presets_data
    try:
        prep_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
        if os.path.exists(prep_path):
            preprocessor = IDSPreprocessor.load(prep_path)
            print("Preprocessor loaded successfully.")
            # Update service options in FEATURE_METADATA
            if 'service' in preprocessor.categories_:
                FEATURE_METADATA['service']['options'] = preprocessor.categories_['service']
                
        model_path = os.path.join(MODELS_DIR, 'tuned_xgb.json')
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            explainer = shap.TreeExplainer(model)
            print("XGBoost model and SHAP explainer loaded successfully.")

        results_path = os.path.join(REPORTS_DIR, 'final_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                master_results = json.load(f)
            optimal_threshold = master_results.get('optimal_threshold', master_results.get('operating_threshold', 0.10))
            print(f"Optimal threshold loaded: {optimal_threshold}")

        preset_path = os.path.join(REPORTS_DIR, 'presets.json')
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as f:
                presets_data = json.load(f)
            print(f"Loaded {len(presets_data)} presets.")
    except Exception as e:
        print(f"Artifact loading note: {e}")

# Load artifacts on startup
load_artifacts()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    if preprocessor and 'service' in preprocessor.categories_:
        FEATURE_METADATA['service']['options'] = preprocessor.categories_['service']
    return jsonify({
        'features': FEATURE_METADATA,
        'all_feature_names': ALL_FEATURES,
        'optimal_threshold': optimal_threshold
    })

@app.route('/api/presets', methods=['GET'])
def get_presets():
    global presets_data
    if not presets_data:
        preset_path = os.path.join(REPORTS_DIR, 'presets.json')
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as f:
                presets_data = json.load(f)
    return jsonify(presets_data)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    global master_results
    if not master_results:
        results_path = os.path.join(REPORTS_DIR, 'final_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                master_results = json.load(f)
    return jsonify(master_results)

@app.route('/api/novel_attacks', methods=['GET'])
def get_novel_attacks():
    novel_path = os.path.join(REPORTS_DIR, 'novel_attack_evaluation.json')
    if os.path.exists(novel_path):
        with open(novel_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/api/figures/<filename>')
def get_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)

@app.route('/api/research_evolution', methods=['GET'])
def get_research_evolution():
    data = {
        "hero": {
            "title": "From Baseline to Robust Intrusion Detection",
            "subtitle": "Experimental progression of XGBoost regularization and cost-sensitive operating-point optimization on NSL-KDD.",
            "progression": [
                {"stage": "Baseline", "accuracy": 80.47, "precision": 96.93, "recall": 67.85, "f1": 0.7982},
                {"stage": "Regularization", "accuracy": 81.80, "precision": 96.90, "recall": 70.28, "f1": 0.8147},
                {"stage": "Final Cost-Sensitive", "accuracy": 86.16, "precision": 96.80, "recall": 78.27, "f1": 0.8655}
            ],
            "improvement_pp": 5.69,
            "final_precision": 96.80,
            "final_accuracy": 86.16
        },
        "experiments": {
            "baseline": {
                "name": "Baseline XGBoost",
                "accuracy": 80.47,
                "precision": 96.93,
                "recall": 67.85,
                "f1": 0.7982,
                "roc_auc": 0.9683,
                "pr_auc": 0.9673,
                "specificity": 97.16,
                "balanced_accuracy": 82.50,
                "fpr": 2.84,
                "fnr": 32.15,
                "threshold": 0.41,
                "novel_attack_detection_rate": 41.44,
                "novel_attacks_detected": 1554,
                "novel_attacks_total": 3750,
                "confusion_matrix": {
                    "tn": 9435,
                    "fp": 276,
                    "fn": 4126,
                    "tp": 8707
                },
                "categories": {
                    "DoS": 87.10,
                    "Probe": 77.70,
                    "R2L": 10.98,
                    "U2R": 19.40,
                    "normal": 97.16
                }
            },
            "regularized": {
                "name": "Regularized Model",
                "accuracy": 81.80,
                "precision": 96.90,
                "recall": 70.28,
                "f1": 0.8147,
                "fpr": 2.91,
                "fnr": 29.72,
                "threshold": 0.36,
                "novel_attack_detection_rate": 48.27,
                "novel_attacks_detected": 1810,
                "novel_attacks_total": 3750
            },
            "final_cost_sensitive": {
                "name": "Final Cost-Sensitive Model",
                "accuracy": 86.16,
                "precision": 96.80,
                "recall": 78.27,
                "f1": 0.8655,
                "roc_auc": 0.9618,
                "pr_auc": 0.9594,
                "specificity": 96.58,
                "balanced_accuracy": 87.42,
                "fpr": 3.42,
                "fnr": 21.73,
                "threshold": 0.10,
                "novel_attack_detection_rate": 70.83,
                "novel_attacks_detected": 2656,
                "novel_attacks_total": 3750,
                "confusion_matrix": {
                    "tn": 9379,
                    "fp": 332,
                    "fn": 2789,
                    "tp": 10044
                },
                "categories": {
                    "DoS": 95.79,
                    "Probe": 94.47,
                    "R2L": 19.99,
                    "U2R": 53.73,
                    "normal": 96.58
                }
            },
            "high_detection_variant": {
                "name": "High Detection Variant",
                "accuracy": 86.14,
                "precision": 93.38,
                "recall": 81.42,
                "f1": 0.8699,
                "fpr": 7.63,
                "fnr": 18.58,
                "threshold": 0.06,
                "novel_attack_detection_rate": 76.99,
                "novel_attacks_detected": 2887,
                "novel_attacks_total": 3750,
                "confusion_matrix": {
                    "tn": 8970,
                    "fp": 741,
                    "fn": 2384,
                    "tp": 10449
                }
            }
        },
        "methodology": [
            {
                "id": 1,
                "step": "Baseline XGBoost",
                "why": "Establish an initial benchmark using standard gradient boosted decision trees with broad parameter defaults on KDDTrain+.",
                "change": "Trained initial XGBoost model with max_depth=6, reg_alpha=1.0, reg_lambda=2.0, evaluated at validation-optimal threshold \u03c4=0.41.",
                "result": "Validation accuracy exceeded 99%, but test accuracy on KDDTest+ was only 80.47% with 32.15% false negative rate, revealing an 19.45 pp generalization gap."
            },
            {
                "id": 2,
                "step": "Generalization Gap Analysis",
                "why": "Diagnose the root cause of the steep performance drop between validation (>99%) and KDDTest+ (80.47%), where 4,126 attacks were missed.",
                "change": "Audited feature distributions, tree depth usage, and KDDTest+ attack types; identified 17 novel attack types absent from training data and distribution shift.",
                "result": "Discovered that deeper trees (depth 6) were memorizing specific training artifacts; recognized necessity of shallow-tree regularization and cost-sensitive thresholding."
            },
            {
                "id": 3,
                "step": "Shallow-tree Regularization",
                "why": "Constrain model complexity to prevent over-specialization to KDDTrain+ and encourage learning of invariant traffic representations.",
                "change": "Reduced max_depth from 6 to 4; increased reg_alpha (1.0 \u2192 3.0), reg_lambda (2.0 \u2192 5.0), min_child_weight (1 \u2192 3), and gamma (0.2 \u2192 0.5).",
                "result": "Untouched test accuracy improved from 80.47% to 81.80% (+1.33 pp), recall increased from 67.85% to 70.28%, and novel attack detection rose from 41.44% to 48.27%."
            },
            {
                "id": 4,
                "step": "Cost-Sensitive Threshold Optimization",
                "why": "In intrusion detection, false negatives (missed attacks) carry severe security consequences compared to false positives (investigation alerts). Standard 0.50 threshold is suboptimal under asymmetric misclassification costs.",
                "change": "Conducted a rigorous decision threshold sweep \u03c4 \u2208 [0.01, 0.99] exclusively on validation data using an asymmetric cost ratio (C_FN:C_FP = 5:1).",
                "result": "Validation optimization identified \u03c4* = 0.10 as the optimal balanced operating point, dramatically reducing missed attacks while keeping false positive rate strictly under 3.5%."
            },
            {
                "id": 5,
                "step": "Untouched Test Evaluation",
                "why": "Assess true real-world generalization without any data snooping, post-hoc threshold adjustment, or test set leakage.",
                "change": "Applied the frozen regularized model at \u03c4* = 0.10 onto the completely untouched KDDTest+ dataset (22,544 instances).",
                "result": "Accuracy reached 86.16% (+5.69 pp over baseline), recall surged from 67.85% to 78.27% (+10.42 pp), while precision remained exceptionally high at 96.80% (F1 = 0.8655)."
            },
            {
                "id": 6,
                "step": "Novel Attack Analysis",
                "why": "Rigorously verify whether the regularized model and cost-sensitive threshold generalized to novel/unseen attack types that were completely absent from training.",
                "change": "Evaluated detection rates specifically on the 3,750 test samples comprising 17 unseen attack types across DoS, Probe, R2L, and U2R.",
                "result": "Novel attack detection rate increased from 41.44% (1,554 / 3,750) to 70.83% (2,656 / 3,750), detecting +1,102 additional novel attacks over baseline."
            }
        ],
        "model_config": {
            "parameters": {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.15,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "min_child_weight": 3,
                "gamma": 0.5,
                "reg_alpha": 3.0,
                "reg_lambda": 5.0,
                "scale_pos_weight": 1.15,
                "operating_threshold": 0.10
            },
            "parameter_changes": [
                {
                    "param": "max_depth",
                    "baseline": 6,
                    "final": 4,
                    "purpose": "Reduce tree complexity / improve generalization by capping hierarchical depth, preventing memorization of training-specific port/IP patterns."
                },
                {
                    "param": "reg_alpha",
                    "baseline": 1.0,
                    "final": 3.0,
                    "purpose": "Stronger L1 regularization on leaf weights, encouraging weight sparsity and pruning uninformative connection noise."
                },
                {
                    "param": "reg_lambda",
                    "baseline": 2.0,
                    "final": 5.0,
                    "purpose": "Stronger L2 regularization on leaf weights, preventing extreme prediction scores and stabilizing tree ensembles."
                },
                {
                    "param": "min_child_weight",
                    "baseline": 1,
                    "final": 3,
                    "purpose": "Discourage overly specific splits by requiring a higher minimum sum of instance weights in each terminal leaf."
                },
                {
                    "param": "gamma",
                    "baseline": 0.2,
                    "final": 0.5,
                    "purpose": "Require stronger split improvement (higher minimum loss reduction) to create a new tree partition."
                }
            ]
        },
        "findings": [
            {"title": "Accuracy", "baseline": "80.47%", "final": "86.16%", "diff": "+5.69 pp", "note": "Substantial gain across all test traffic"},
            {"title": "Recall", "baseline": "67.85%", "final": "78.27%", "diff": "+10.42 pp", "note": "Significantly higher threat detection"},
            {"title": "F1-Score", "baseline": "0.7982", "final": "0.8655", "diff": "+0.0673", "note": "Optimal harmonic balance"},
            {"title": "Novel Attack Detection", "baseline": "41.44%", "final": "70.83%", "diff": "+29.39 pp", "note": "+1,102 additional novel attacks detected"},
            {"title": "False Negatives", "baseline": "4,126", "final": "2,789", "diff": "-1,337 missed", "note": "1,337 fewer missed attacks"},
            {"title": "Precision", "baseline": "96.93%", "final": "96.80%", "diff": "-0.13 pp", "note": "Remained reliably above 96%"}
        ],
        "integrity_checklist": [
            "KDDTrain+ \u2192 80/20 stratified split (100,778 train / 25,195 validation)",
            "Preprocessor fitted only on training data (no feature or scaling leakage)",
            "Validation used for threshold selection (\u03c4* = 0.10 chosen before test evaluation)",
            "KDDTest+ untouched until final evaluation (22,544 rows evaluated strictly once)",
            "17 novel attack types absent from training (3,750 novel attack instances)",
            "Final metrics independently reproduced and verified across all project reports"
        ]
    }
    return jsonify(data)

@app.route('/api/predict', methods=['POST'])
def predict():
    global preprocessor, model, explainer, optimal_threshold
    if model is None or preprocessor is None:
        load_artifacts()
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model or preprocessor not yet loaded. Please wait for pipeline to complete.'}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Validate input features
        record = {}
        for col in ALL_FEATURES:
            if col not in data:
                # Use default if missing
                record[col] = FEATURE_METADATA.get(col, {}).get('default', 0)
            else:
                val = data[col]
                # Convert numeric/binary
                if col in NUMERICAL_FEATURES:
                    try:
                        record[col] = float(val)
                    except ValueError:
                        return jsonify({'error': f"Invalid numeric value for '{col}': {val}"}), 400
                else:
                    record[col] = str(val)

        # Allow user to specify custom threshold if desired
        custom_thresh = request.args.get('threshold', None)
        threshold_to_use = float(custom_thresh) if custom_thresh is not None else optimal_threshold

        # Preprocess and Predict
        df_single = pd.DataFrame([record])
        X_trans = preprocessor.transform(df_single)
        
        prob = float(model.predict_proba(X_trans)[0, 1])
        is_attack = bool(prob >= threshold_to_use)
        
        confidence = prob if is_attack else (1.0 - prob)

        # Compute SHAP explanation for this individual connection
        explanation = explain_single_instance(explainer, model, preprocessor, df_single, top_k=5)

        return jsonify({
            'prediction': 'MALICIOUS TRAFFIC' if is_attack else 'NORMAL TRAFFIC',
            'is_attack': is_attack,
            'attack_probability': round(prob * 100, 2),
            'confidence': round(confidence * 100, 2),
            'threshold_used': round(threshold_to_use, 4),
            'status': '⚠ Potential intrusion detected' if is_attack else 'No intrusion detected',
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    global preprocessor, model, optimal_threshold
    if model is None or preprocessor is None:
        load_artifacts()
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not ready'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        df_batch = pd.read_csv(file)
        
        # Check if features are present
        missing = [c for c in ALL_FEATURES if c not in df_batch.columns]
        if missing:
            return jsonify({'error': f"CSV missing required columns: {missing[:5]}..."}), 400

        X_trans = preprocessor.transform(df_batch)
        probs = model.predict_proba(X_trans)[:, 1]
        preds = (probs >= optimal_threshold).astype(int)

        results = []
        has_actual = LABEL_COLUMN in df_batch.columns
        
        for idx in range(min(500, len(df_batch))):
            row_res = {
                'row_index': idx + 1,
                'prediction': 'MALICIOUS' if preds[idx] == 1 else 'NORMAL',
                'attack_probability': round(float(probs[idx]) * 100, 2),
            }
            if has_actual:
                actual_label = str(df_batch[LABEL_COLUMN].iloc[idx])
                is_actual_attack = actual_label != 'normal'
                is_correct = bool((preds[idx] == 1) == is_actual_attack)
                row_res['actual_label'] = actual_label
                row_res['is_correct'] = is_correct
            results.append(row_res)

        total_rows = len(df_batch)
        total_attacks = int((preds == 1).sum())
        total_normal = total_rows - total_attacks

        return jsonify({
            'total_rows': total_rows,
            'total_attacks_flagged': total_attacks,
            'total_normal_flagged': total_normal,
            'threshold_used': optimal_threshold,
            'sample_predictions': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream_sample', methods=['GET'])
def get_stream_sample():
    """
    Returns a sample of real records from KDDTest+ for the live simulation stream.
    """
    try:
        df_test = pd.read_csv(TEST_DATA_PATH, names=RAW_COLUMNS, header=None)
        # Sample 60 connections (mix of normal and various attack types)
        sample = df_test.sample(n=60, random_state=np.random.randint(1, 10000))
        
        records = []
        for _, row in sample.iterrows():
            rec = row.to_dict()
            # Clean types
            clean_rec = {}
            for k, v in rec.items():
                if isinstance(v, (np.integer, int)):
                    clean_rec[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_rec[k] = float(v)
                else:
                    clean_rec[k] = str(v)
            records.append(clean_rec)
            
        return jsonify(records)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
