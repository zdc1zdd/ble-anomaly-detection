"""
features.py
滑动窗口特征抽出
"""

import pandas as pd
import numpy as np
import config


def extract_window_features(rssi_series, window_size=None):
    """
    对每个滑动窗口抽出统计特征
    输出窗口数 = len(rssi_series) - window_size + 1
    每个窗口的标签 = 窗口最后一个采样点的标签 (因果性,只用过去的信息)
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
    
    rssi = rssi_series.values if hasattr(rssi_series, 'values') else np.asarray(rssi_series)
    n = len(rssi)
    
    features = {
        'mean':  np.zeros(n - window_size + 1),
        'std':   np.zeros(n - window_size + 1),
        'min':   np.zeros(n - window_size + 1),
        'max':   np.zeros(n - window_size + 1),
        'range': np.zeros(n - window_size + 1),
        'last':  np.zeros(n - window_size + 1),
        'slope': np.zeros(n - window_size + 1),
    }
    
    for i in range(window_size, n + 1):
        win = rssi[i - window_size:i]
        idx = i - window_size
        features['mean'][idx]  = win.mean()
        features['std'][idx]   = win.std()
        features['min'][idx]   = win.min()
        features['max'][idx]   = win.max()
        features['range'][idx] = win.max() - win.min()
        features['last'][idx]  = win[-1]
        features['slope'][idx] = win[-1] - win[0]
    
    return pd.DataFrame(features)


def prepare_train_test(df_full, window_size=None):
    """
    从prepare_data()的输出 → 训练/测试用的(X, y)
    
    返回 dict:
        X_train, y_train, X_test, y_test : ndarray
        anom_type_test : 测试集每个样本的异常类型 (用于breakdown评价)
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
    
    # 抽出特征
    X_all = extract_window_features(df_full['rssi'], window_size)
    
    # 对应的label/split/anomaly_type (窗口最后一点)
    y_all = df_full['label'].iloc[window_size - 1:].values
    split_all = df_full['split'].iloc[window_size - 1:].values
    type_all = df_full['anomaly_type'].iloc[window_size - 1:].values
    
    train_mask = split_all == 'train'
    test_mask = split_all == 'test'
    
    return {
        'X_train': X_all[train_mask].values,
        'y_train': y_all[train_mask],
        'X_test': X_all[test_mask].values,
        'y_test': y_all[test_mask],
        'anom_type_test': type_all[test_mask],
        'feature_names': list(X_all.columns),
    }


if __name__ == '__main__':
    import data_loader
    df = data_loader.prepare_data()
    data = prepare_train_test(df)
    print(f"X_train: {data['X_train'].shape}, y_train: {data['y_train'].shape}")
    print(f"X_test: {data['X_test'].shape}, y_test: {data['y_test'].shape}")
    print(f"训练集异常率: {data['y_train'].mean()*100:.2f}%")
    print(f"测试集异常率: {data['y_test'].mean()*100:.2f}%")
    print(f"\n测试集中各异常类型样本数:")
    types, counts = np.unique(data['anom_type_test'], return_counts=True)
    for t, c in zip(types, counts):
        n_anom = ((data['anom_type_test'] == t) & (data['y_test'] == 1)).sum()
        print(f"  {t}: 总数{c}, 异常{n_anom}")
