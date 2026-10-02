import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')

# Dataset file paths
TRAIN_DATA_PATH = os.path.join(DATASET_DIR, 'KDDTrain+.txt')
TEST_DATA_PATH = os.path.join(DATASET_DIR, 'KDDTest+.txt')

# Random seed for reproducibility
RANDOM_STATE = 42

# Complete raw NSL-KDD column definitions (43 columns)
RAW_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

# Zero-variance column to drop with technical justification
DROP_COLUMNS = ['num_outbound_cmds']

# Benchmark metadata column to drop to prevent target/evaluation leakage
METADATA_COLUMNS = ['difficulty_level']

# Categorical features
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

# Active numerical features (37 features)
NUMERICAL_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Complete list of 40 active predictive features in order
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Binary target mapping: 0 = Normal, 1 = Attack
LABEL_COLUMN = 'attack_type'
TARGET_COLUMN = 'binary_target'

# NSL-KDD Attack Categories mapping
ATTACK_CATEGORIES = {
    # DoS
    'apache2': 'DoS', 'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS',
    'mailbomb': 'DoS', 'pod': 'DoS', 'processtable': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'mscan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'saint': 'Probe', 'satan': 'Probe',
    # R2L (Remote to Local)
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'httptunnel': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L', 'sendmail': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
    'warezmaster': 'R2L', 'worm': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    # U2R (User to Root)
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'ps': 'U2R',
    'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
    # Normal
    'normal': 'normal'
}

# Plain-English cybersecurity explanations for user interface & tooltips
FEATURE_METADATA = {
    # Group 1: Basic Connection Information
    'protocol_type': {
        'group': 'Group 1 — Basic Connection',
        'label': 'Protocol Type',
        'type': 'categorical',
        'options': ['tcp', 'udp', 'icmp'],
        'default': 'tcp',
        'description': 'Specifies the network protocol used by the connection (TCP, UDP, or ICMP).'
    },
    'service': {
        'group': 'Group 1 — Basic Connection',
        'label': 'Network Service',
        'type': 'categorical',
        'options': [],  # Dynamically populated from training data
        'default': 'http',
        'description': 'Specifies the network service/port requested on the destination host (e.g., HTTP, FTP, SMTP, SSH).'
    },
    'flag': {
        'group': 'Group 1 — Basic Connection',
        'label': 'Connection Status Flag',
        'type': 'categorical',
        'options': ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH'],
        'default': 'SF',
        'description': 'Represents the connection completion or error state (e.g., SF=Normal establishment/termination, S0=Connection attempt seen with no reply, REJ=Connection rejected).'
    },
    'duration': {
        'group': 'Group 1 — Basic Connection',
        'label': 'Duration (seconds)',
        'type': 'numeric',
        'min': 0, 'max': 100000, 'step': 1, 'default': 0,
        'description': 'Duration of the connection in seconds from start to termination.'
    },

    # Group 2: Data Transfer
    'src_bytes': {
        'group': 'Group 2 — Data Transfer',
        'label': 'Source Bytes',
        'type': 'numeric',
        'min': 0, 'max': 1500000000, 'step': 1, 'default': 294,
        'description': 'Number of data bytes transmitted from source host to destination host.'
    },
    'dst_bytes': {
        'group': 'Group 2 — Data Transfer',
        'label': 'Destination Bytes',
        'type': 'numeric',
        'min': 0, 'max': 1500000000, 'step': 1, 'default': 6442,
        'description': 'Number of data bytes transmitted back from destination host to source host.'
    },

    # Group 3: Behaviour / Login Information
    'num_failed_logins': {
        'group': 'Group 3 — Behaviour / Login',
        'label': 'Failed Logins',
        'type': 'numeric',
        'min': 0, 'max': 10, 'step': 1, 'default': 0,
        'description': 'Number of failed login attempts during this connection.'
    },
    'logged_in': {
        'group': 'Group 3 — Behaviour / Login',
        'label': 'Logged In',
        'type': 'binary',
        'default': 1,
        'description': 'Indicates whether the connection resulted in a successfully authenticated session (1=Yes, 0=No).'
    },
    'count': {
        'group': 'Group 3 — Behaviour / Login',
        'label': 'Same-Host Connection Count',
        'type': 'numeric',
        'min': 0, 'max': 1000, 'step': 1, 'default': 22,
        'description': 'Number of connections to the same destination host in the past 2-second observation window.'
    },
    'srv_count': {
        'group': 'Group 3 — Behaviour / Login',
        'label': 'Same-Service Connection Count',
        'type': 'numeric',
        'min': 0, 'max': 1000, 'step': 1, 'default': 46,
        'description': 'Number of connections to the same service in the past 2-second observation window.'
    },
    'num_compromised': {
        'group': 'Group 3 — Behaviour / Login',
        'label': 'Compromised Conditions',
        'type': 'numeric',
        'min': 0, 'max': 1000, 'step': 1, 'default': 0,
        'description': 'Number of compromised conditions/states observed on the system.'
    },
    'root_shell': {
        'group': 'Group 3 — Behaviour / Login',
        'label': 'Root Shell Obtained',
        'type': 'binary',
        'default': 0,
        'description': 'Indicates whether root/administrator shell access was obtained (1=Yes, 0=No).'
    },

    # Group 4: Error Rate Information
    'serror_rate': {
        'group': 'Group 4 — Error Rates',
        'label': 'SYN Error Rate (Same Host)',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'Proportion of connections to the same host that encountered SYN errors (0.0 to 1.0).'
    },
    'srv_serror_rate': {
        'group': 'Group 4 — Error Rates',
        'label': 'SYN Error Rate (Same Service)',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'Proportion of connections to the same service that encountered SYN errors (0.0 to 1.0).'
    },
    'rerror_rate': {
        'group': 'Group 4 — Error Rates',
        'label': 'REJ Error Rate (Same Host)',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'Proportion of connections to the same host that encountered REJ (rejected) errors (0.0 to 1.0).'
    },
    'srv_rerror_rate': {
        'group': 'Group 4 — Error Rates',
        'label': 'REJ Error Rate (Same Service)',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'Proportion of connections to the same service that encountered REJ (rejected) errors (0.0 to 1.0).'
    },
    'same_srv_rate': {
        'group': 'Group 4 — Error Rates',
        'label': 'Same Service Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 1.0,
        'description': 'Proportion of connections that were to the same service (0.0 to 1.0).'
    },
    'diff_srv_rate': {
        'group': 'Group 4 — Error Rates',
        'label': 'Different Service Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'Proportion of connections that were to different services (0.0 to 1.0).'
    },
    'srv_diff_host_rate': {
        'group': 'Group 4 — Error Rates',
        'label': 'Service Different Host Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.11,
        'description': 'Proportion of connections to the same service that were to different hosts (0.0 to 1.0).'
    },

    # Group 5: Connection / Security Indicators
    'land': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Land Attack Flag',
        'type': 'binary',
        'default': 0,
        'description': 'Indicates if source and destination IP/port addresses are identical (1=Yes, spoofed loopback attack, 0=No).'
    },
    'wrong_fragment': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Wrong Fragments',
        'type': 'numeric',
        'min': 0, 'max': 10, 'step': 1, 'default': 0,
        'description': 'Number of malformed or overlapping packet fragments received.'
    },
    'urgent': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Urgent Packets',
        'type': 'numeric',
        'min': 0, 'max': 10, 'step': 1, 'default': 0,
        'description': 'Number of urgent packets in this connection.'
    },
    'hot': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Hot Indicators',
        'type': 'numeric',
        'min': 0, 'max': 100, 'step': 1, 'default': 0,
        'description': 'Number of "hot" indicators (e.g., accessing system directories, executing privileged scripts).'
    },
    'su_attempted': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Superuser Attempted',
        'type': 'numeric',
        'min': 0, 'max': 2, 'step': 1, 'default': 0,
        'description': 'Indicates whether "su root" command was attempted (1=Attempted, 2=Successful, 0=No).'
    },
    'num_root': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Root Operations Count',
        'type': 'numeric',
        'min': 0, 'max': 1000, 'step': 1, 'default': 0,
        'description': 'Number of operations performed with root privileges.'
    },
    'num_file_creations': {
        'group': 'Group 5 — Security Indicators',
        'label': 'File Creations',
        'type': 'numeric',
        'min': 0, 'max': 100, 'step': 1, 'default': 0,
        'description': 'Number of file creation operations during this connection.'
    },
    'num_shells': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Shell Prompts Spawned',
        'type': 'numeric',
        'min': 0, 'max': 10, 'step': 1, 'default': 0,
        'description': 'Number of shell command prompt sessions spawned.'
    },
    'num_access_files': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Access Control Files Touched',
        'type': 'numeric',
        'min': 0, 'max': 10, 'step': 1, 'default': 0,
        'description': 'Number of operations on access control files (e.g., /etc/passwd, .rhosts).'
    },
    'is_host_login': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Host Login',
        'type': 'binary',
        'default': 0,
        'description': 'Indicates whether login was directly to the host machine (1=Yes, 0=No).'
    },
    'is_guest_login': {
        'group': 'Group 5 — Security Indicators',
        'label': 'Guest Login',
        'type': 'binary',
        'default': 0,
        'description': 'Indicates whether the connection logged in using a guest account (1=Yes, 0=No).'
    },

    # Group 6: Destination Host Statistics
    'dst_host_count': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host Count',
        'type': 'numeric',
        'min': 0, 'max': 255, 'step': 1, 'default': 180,
        'description': 'Count of connections having the same destination host IP address (window up to 255).'
    },
    'dst_host_srv_count': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host Service Count',
        'type': 'numeric',
        'min': 0, 'max': 255, 'step': 1, 'default': 255,
        'description': 'Count of connections having the same destination host IP address and port/service.'
    },
    'dst_host_same_srv_rate': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host Same Service Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 1.0,
        'description': 'Proportion of connections to the destination host that use the same service (0.0 to 1.0).'
    },
    'dst_host_diff_srv_rate': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host Diff Service Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'Proportion of connections to the destination host that use different services (0.0 to 1.0).'
    },
    'dst_host_same_src_port_rate': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host Same Source Port Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.01,
        'description': 'Proportion of connections to the destination host coming from the same source port (0.0 to 1.0).'
    },
    'dst_host_srv_diff_host_rate': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host Service Diff Host Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.01,
        'description': 'Proportion of connections to the same service that originated from different hosts (0.0 to 1.0).'
    },
    'dst_host_serror_rate': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host SYN Error Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'SYN error rate among connections to the destination host (0.0 to 1.0).'
    },
    'dst_host_srv_serror_rate': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host Service SYN Error Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'SYN error rate among connections to the same service on the destination host (0.0 to 1.0).'
    },
    'dst_host_rerror_rate': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host REJ Error Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'Rejected connection rate among connections to the destination host (0.0 to 1.0).'
    },
    'dst_host_srv_rerror_rate': {
        'group': 'Group 6 — Destination Host Stats',
        'label': 'Dest Host Service REJ Error Rate',
        'type': 'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 0.0,
        'description': 'Rejected connection rate among connections to the same service on the destination host (0.0 to 1.0).'
    }
}
