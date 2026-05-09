"""
main_pipeline.py
完整pipeline: 数据加载 → 前处理 → 特征 → 训练 → 评价
跑一遍出基线结果(默认超参数)
"""

import os
import pandas as pd

import config
import data_loader
import features
import methods
import evaluator


def run_pipeline(verbose=True):
    """
    完整pipeline
    返回:
        results: {method_name: evaluate()输出} 各手法的整体评价
        results_by_type: {method_name: evaluate_by_anomaly_type()输出} 异常类型breakdown
        data: 训练/测试用的数据 (X_train, y_train, X_test, y_test, etc.)
    """
    # ==========================================
    # Step 1: 数据准备
    # ==========================================
    if verbose:
        print("="*60)
        print("Step 1: 数据加载与前处理")
        print("="*60)
    
    df = data_loader.prepare_data()
    
    if verbose:
        print(f"数据形状: {df.shape}")
        print(f"插补方法: {config.INTERPOLATION_METHOD}")
        print(f"训练集: {(df['split']=='train').sum()}秒, "
              f"異常率 {df[df['split']=='train']['label'].mean()*100:.2f}%")
        print(f"测试集: {(df['split']=='test').sum()}秒, "
              f"異常率 {df[df['split']=='test']['label'].mean()*100:.2f}%")
    
    # ==========================================
    # Step 2: 特征抽出
    # ==========================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 2: 特征抽出 (窗口大小 W={config.WINDOW_SIZE}秒)")
        print("="*60)
    
    data = features.prepare_train_test(df)
    
    if verbose:
        print(f"特征数: {data['X_train'].shape[1]}")
        print(f"特征名: {data['feature_names']}")
        print(f"训练样本: {len(data['X_train'])}, 测试样本: {len(data['X_test'])}")
    
    # ==========================================
    # Step 3: 各手法训练 + 预测
    # ==========================================
    if verbose:
        print(f"\n{'='*60}")
        print("Step 3: 各手法训练与预测")
        print("="*60)
    
    method_list = methods.get_default_methods()
    predictions = {}
    
    for method in method_list:
        method.fit(data['X_train'], data['y_train'])
        y_pred = method.predict(data['X_test'])
        predictions[method.name] = y_pred
        if verbose:
            print(f"  ✓ {method.name}: 预测异常数 {y_pred.sum()}")
    
    # ==========================================
    # Step 4: 整体评价
    # ==========================================
    if verbose:
        print(f"\n{'='*60}")
        print("Step 4: 整体性能评价 (検出率 / 誤報率)")
        print("="*60)
    
    results = {}
    for name, y_pred in predictions.items():
        results[name] = evaluator.evaluate(data['y_test'], y_pred)
    
    summary_table = evaluator.format_results_table(results)
    if verbose:
        print(summary_table.to_string(index=False))
    
    # ==========================================
    # Step 5: 按异常类型breakdown评价
    # ==========================================
    if verbose:
        print(f"\n{'='*60}")
        print("Step 5: 按异常类型的检出率breakdown")
        print("="*60)
    
    results_by_type = {}
    for name, y_pred in predictions.items():
        results_by_type[name] = evaluator.evaluate_by_anomaly_type(
            data['y_test'], y_pred, data['anom_type_test']
        )
    
    if verbose:
        # 整理成一个透视表
        rows = []
        for name, df_type in results_by_type.items():
            for _, r in df_type.iterrows():
                rows.append({
                    '手法': name,
                    '異常類型': r['anomaly_type'],
                    '検出率 [%]': r['detection_rate'] * 100,
                })
        breakdown_df = pd.DataFrame(rows)
        pivot = breakdown_df.pivot(index='手法', columns='異常類型', values='検出率 [%]')
        # 保持手法的顺序 (按method_list)
        pivot = pivot.reindex([m.name for m in method_list])
        print(pivot.round(2).to_string())
    
    return {
        'results': results,
        'results_by_type': results_by_type,
        'data': data,
        'predictions': predictions,
        'summary_table': summary_table,
    }


if __name__ == '__main__':
    output = run_pipeline(verbose=True)
    
    # 保存结果
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output['summary_table'].to_csv(
        f"{config.OUTPUT_DIR}/baseline_results.csv", index=False, encoding='utf-8'
    )
    print(f"\n结果保存到 {config.OUTPUT_DIR}/baseline_results.csv")
