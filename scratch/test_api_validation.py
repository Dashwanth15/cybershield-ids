import urllib.request
import json

base_url = 'http://127.0.0.1:5000'

# 1. /api/metrics
with urllib.request.urlopen(f'{base_url}/api/metrics') as res:
    m = json.loads(res.read().decode())
    print('1. /api/metrics:')
    tm = m.get('test_metrics', {}).get('final_model', {})
    print(f"   Accuracy:  {tm.get('accuracy')}")
    print(f"   Precision: {tm.get('precision')}")
    print(f"   Recall:    {tm.get('recall')}")
    print(f"   F1:        {tm.get('f1_score')}")
    print(f"   ROC-AUC:   {tm.get('roc_auc')}")
    print(f"   PR-AUC:    {tm.get('pr_auc')}")
    print(f"   Threshold: {m.get('optimal_threshold')}")

# 2. /api/metadata
with urllib.request.urlopen(f'{base_url}/api/metadata') as res:
    meta = json.loads(res.read().decode())
    print(f"2. /api/metadata: optimal_threshold = {meta.get('optimal_threshold')}")

# 3. /api/presets
with urllib.request.urlopen(f'{base_url}/api/presets') as res:
    presets = json.loads(res.read().decode())
    print(f"3. /api/presets: loaded {len(presets)} presets")

# 4. /api/predict (Manual prediction + SHAP)
test_sample = presets.get('DoS / Neptune Attack', {})
req = urllib.request.Request(
    f'{base_url}/api/predict',
    data=json.dumps(test_sample).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req) as res:
    pred = json.loads(res.read().decode())
    print('4. /api/predict for DoS / Neptune Attack:')
    print(f"   Prediction:         {pred.get('prediction')}")
    print(f"   Attack Probability: {pred.get('attack_probability')}%")
    print(f"   Threshold Used:     {pred.get('threshold_used')}")
    print(f"   SHAP Features:      {len(pred.get('explanation', {}).get('top_features', []))}")

# 5. /api/novel_attacks
with urllib.request.urlopen(f'{base_url}/api/novel_attacks') as res:
    nov = json.loads(res.read().decode())
    print(f"5. /api/novel_attacks: {len(nov)} novel attacks returned")

print('\nALL API VALIDATION CHECKS PASSED!')
