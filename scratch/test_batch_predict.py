import urllib.request
import io
import json
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_COLUMNS

df = pd.read_csv('dataset/KDDTest+.txt', header=None, nrows=10, names=RAW_COLUMNS)
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)

boundary = '----WebKitFormBoundaryTest12345'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test_batch.csv"\r\n'
    f'Content-Type: text/csv\r\n\r\n'
    f'{csv_buf.getvalue()}\r\n'
    f'--{boundary}--\r\n'
).encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/batch_predict',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

with urllib.request.urlopen(req) as res:
    result = json.loads(res.read().decode())
    print('Batch Predict Success!')
    print('Total Records:', result.get('total_records'))
    print('Attacks Detected:', result.get('attacks_detected'))
    print('Normal Detected:', result.get('normal_detected'))
