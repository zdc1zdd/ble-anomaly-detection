"""
data_loader.py
数据加载、等间隔化、插补、5分钟分块分割
"""

import pandas as pd
import numpy as np
import config


def load_raw(path=None):
    """读取原始log文件"""
    if path is None:
        path = config.DATA_PATH
    df = pd.read_csv(path, header=None, names=config.COLUMNS)
    df['datetime'] = pd.to_datetime(df['unixtime'], unit='s')
    df = df.set_index('datetime')
    return df


def kalman_interpolate(y, Q=0.5, R=2.0):
    """
    简易卡尔曼滤波插补
    モデル: x_{t+1} = x_t + w,  w ~ N(0, Q)
    観測:   y_t = x_t + v,    v ~ N(0, R)
    """
    n = len(y)
    x_hat = np.zeros(n)
    P = np.zeros(n)
    
    first_valid = y.first_valid_index()
    first_idx = y.index.get_loc(first_valid)
    x_hat[first_idx] = y.loc[first_valid]
    P[first_idx] = R
    
    for i in range(first_idx + 1, n):
        x_pred = x_hat[i-1]
        P_pred = P[i-1] + Q
        if not np.isnan(y.iloc[i]):
            K = P_pred / (P_pred + R)
            x_hat[i] = x_pred + K * (y.iloc[i] - x_pred)
            P[i] = (1 - K) * P_pred
        else:
            x_hat[i] = x_pred
            P[i] = P_pred
    
    return pd.Series(x_hat, index=y.index)


def resample_and_interpolate(df, freq=None, method=None):
    """
    等间隔化 + 插补
    返回的DataFrame含: rssi, label, t_min(从开始的分钟数)
    """
    if freq is None:
        freq = config.RESAMPLE_FREQ
    if method is None:
        method = config.INTERPOLATION_METHOD
    
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df_full = df[['rssi', 'label']].reindex(full_idx)
    
    # RSSI插补
    if method == 'linear':
        df_full['rssi'] = df_full['rssi'].interpolate(method='linear')
    elif method == 'ffill':
        df_full['rssi'] = df_full['rssi'].ffill()
    elif method == 'kalman':
        df_full['rssi'] = kalman_interpolate(df_full['rssi'])
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # 标签用前值复制
    df_full['label'] = df_full['label'].ffill()
    df_full = df_full.dropna()
    df_full['label'] = df_full['label'].astype(int)
    
    # 添加时间标记(分钟)
    df_full['t_min'] = np.arange(len(df_full)) / 60.0
    
    return df_full


def assign_blocks(df_full, block_size_min=None):
    """
    分配5分钟块,奇偶分配train/test
    返回: 添加了 'block' 和 'split' 两列的DataFrame
    """
    if block_size_min is None:
        block_size_min = config.BLOCK_SIZE_MIN
    
    df = df_full.copy()
    df['block'] = (df['t_min'] // block_size_min).astype(int)
    df['split'] = df['block'].apply(lambda b: 'train' if b % 2 == 0 else 'test')
    return df


def assign_anomaly_types(df_full):
    """根据时刻标记每个时点的异常类型"""
    def label_type(t):
        for start, end, name in config.ANOMALY_TYPE_BOUNDARIES:
            if start <= t < end:
                return name
        return 'Unknown'
    
    df = df_full.copy()
    df['anomaly_type'] = df['t_min'].apply(label_type)
    return df


def prepare_data():
    """
    一站式调用: 加载 → 等间隔化 → 插补 → 分块 → 异常类型标记
    返回处理好的DataFrame
    """
    df_raw = load_raw()
    df_full = resample_and_interpolate(df_raw)
    df_full = assign_blocks(df_full)
    df_full = assign_anomaly_types(df_full)
    return df_full


if __name__ == '__main__':
    # 测试
    df = prepare_data()
    print(f"数据形状: {df.shape}")
    print(f"\n列: {list(df.columns)}")
    print(f"\n训练/测试分布:")
    print(df['split'].value_counts())
    print(f"\n各异常类型分布:")
    print(df['anomaly_type'].value_counts())
    print(f"\nsplit × anomaly_type 交叉表:")
    print(pd.crosstab(df['anomaly_type'], df['split']))
