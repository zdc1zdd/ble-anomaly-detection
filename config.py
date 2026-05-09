"""
config.py
集中管理所有超参数和路径
"""
import os

# ========== 路径 ==========
# 项目根目录(本文件所在目录)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 数据 ==========
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'bleHitachi_Labeled.log')
COLUMNS = ['unixtime', 'device_id', 'rssi', 'temp', 'humid', 'label']

# ========== 前处理 ==========
RESAMPLE_FREQ = '1s'           # 等间隔化的频率
INTERPOLATION_METHOD = 'linear'  # 'linear' / 'ffill' / 'kalman'

# ========== 数据分割 ==========
BLOCK_SIZE_MIN = 5             # 5分钟一块
# 偶数块训练 / 奇数块测试 (block 0,2,4,...→train; 1,3,5,...→test)

# ========== 特征抽出 ==========
WINDOW_SIZE = 30               # 滑动窗口大小(秒)
FEATURE_NAMES = ['mean', 'std', 'min', 'max', 'range', 'last', 'slope']

# ========== 异常类型时间区分 ==========
ANOMALY_TYPE_BOUNDARIES = [
    (0, 150, 'Normal'),
    (150, 180, 'Short'),       # 15s closure
    (180, 210, 'Medium'),      # 30s closure
    (210, 240, 'Long'),        # 60s closure
    (240, 270, 'Gradual'),     # gradual decay
    (270, 300, 'Alternating'), # alternating
]

# ========== 各手法的默认超参数 ==========
HPARAMS = {
    'Baseline': {
        'k': 1.5,              # mu - k*sigma
    },
    'LogisticRegression': {
        'C': 1.0,
        'class_weight': 'balanced',
        'max_iter': 1000,
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': None,
        'class_weight': 'balanced',
        'random_state': 42,
    },
    'IsolationForest': {
        'n_estimators': 100,
        'contamination': 0.07,
        'random_state': 42,
    },
    'OneClassSVM': {
        'nu': 0.07,
        'kernel': 'rbf',
        'gamma': 'scale',
    },
}

# ========== 超参数扫描范围 ==========
HPARAM_GRID = {
    'Baseline_k':         [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    'LogReg_C':           [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'RF_max_depth':       [2, 3, 5, 8, 12, 20, None],
    'IForest_contamination': [0.02, 0.05, 0.07, 0.10, 0.15, 0.20],
    'OCSVM_nu':           [0.02, 0.05, 0.07, 0.10, 0.15, 0.20],
}

# ========== 输出 ==========
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
RANDOM_SEED = 42
