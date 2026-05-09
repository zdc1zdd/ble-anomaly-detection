"""
evaluator.py
检出率と誤報率の計算
"""

import numpy as np
import pandas as pd


def confusion_components(y_true, y_pred):
    """计算 TP, FN, FP, TN"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    return TP, FN, FP, TN


def detection_rate(y_true, y_pred):
    """検出率 = TP / (TP + FN)"""
    TP, FN, _, _ = confusion_components(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def false_alarm_rate(y_true, y_pred):
    """誤報率 = FP / (FP + TN)"""
    _, _, FP, TN = confusion_components(y_true, y_pred)
    return FP / (FP + TN) if (FP + TN) > 0 else 0.0


def evaluate(y_true, y_pred):
    """
    返回包含检出率、误报率、TP/FN/FP/TN的字典
    """
    TP, FN, FP, TN = confusion_components(y_true, y_pred)
    dr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    far = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return {
        'detection_rate': dr,
        'false_alarm_rate': far,
        'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
    }


def evaluate_by_anomaly_type(y_true, y_pred, anom_types):
    """
    按异常类型分组的评价
    各类型的"检出率"= 该类型异常中被检出的比例
    "误报率"统一计算 (因为所有type都共享同一个Normal集合,误报率不分类型)
    
    返回 DataFrame
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    anom_types = np.asarray(anom_types)
    
    rows = []
    
    # 各异常类型的检出率
    for atype in sorted(set(anom_types)):
        if atype == 'Normal':
            continue
        mask = anom_types == atype
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        # 该类型时段中,真实异常的样本数和被检出数
        n_anom = (sub_true == 1).sum()
        n_detected = ((sub_true == 1) & (sub_pred == 1)).sum()
        dr = n_detected / n_anom if n_anom > 0 else 0.0
        # 该时段的样本数(包括正常和异常)
        rows.append({
            'anomaly_type': atype,
            'n_total_in_period': len(sub_true),
            'n_anomaly': int(n_anom),
            'n_detected': int(n_detected),
            'detection_rate': dr,
        })
    
    # 整体误报率 (Normal时段的FP/总Normal数)
    normal_mask = anom_types == 'Normal'
    far = false_alarm_rate(y_true[normal_mask], y_pred[normal_mask])
    
    df = pd.DataFrame(rows)
    df['overall_false_alarm_rate'] = far
    return df


def format_results_table(results_dict):
    """
    将多个手法的结果整理成一个表
    输入: {method_name: evaluate()的输出}
    返回: DataFrame
    """
    rows = []
    for name, res in results_dict.items():
        rows.append({
            '手法': name,
            '検出率 [%]': res['detection_rate'] * 100,
            '誤報率 [%]': res['false_alarm_rate'] * 100,
            'TP': res['TP'],
            'FN': res['FN'],
            'FP': res['FP'],
            'TN': res['TN'],
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # 测试
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    res = evaluate(y_true, y_pred)
    print(res)
    print(f"検出率 = {res['detection_rate']*100:.1f}%")
    print(f"誤報率 = {res['false_alarm_rate']*100:.1f}%")
