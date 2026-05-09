# BLE RSSI异常检测 — 新人研修テーマC

基于BLE传感器观测的RSSI时间序列,比较监督学习与无监督学习的异常检测性能。

## 项目结构

```
ble_anomaly_detection/
├── data/
│   └── bleHitachi_Labeled.log     # 日立提供的原始数据
├── config.py                       # 超参数和路径配置
├── data_loader.py                  # 数据加载、插补、5分钟分块
├── features.py                     # 滑动窗口特征抽出
├── methods.py                      # 5个异常检测手法的封装
├── evaluator.py                    # 检出率/误报率计算
├── main_pipeline.py                # 主pipeline (一键运行)
├── requirements.txt                # 依赖库
└── outputs/                        # 结果输出目录(自动生成)
```

## 环境准备

### 推荐: 用虚拟环境

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 不用虚拟环境(简单)
```
pip install -r requirements.txt
```

需要 Python 3.9 以上。

## 运行方法

### 一键跑全流程
```
python main_pipeline.py
```

预期输出:
```
============================================================
Step 1: 数据加载与前处理
============================================================
数据形状: (17996, 6)
插补方法: linear
训练集: 9000秒, 異常率 6.86%
测试集: 8996秒, 異常率 7.64%

============================================================
Step 2: 特征抽出 (窗口大小 W=30秒)
============================================================
...

============================================================
Step 4: 整体性能评价 (検出率 / 誤報率)
============================================================
             手法    検出率 [%]  誤報率 [%]  TP  FN  FP   TN
       Baseline  83.260553 4.982549 572 115 414 7895
         LogReg  99.126638 1.672885 681   6 139 8170
   RandomForest  95.196507 0.421230 654  33  35 8274
IsolationForest  67.976710 2.370923 467 220 197 8112
    OneClassSVM 100.000000 6.655434 687   0 553 7756
```

### 单独测试各模块

每个模块都可以独立运行,用来检查这个模块工作正常:

```
python data_loader.py    # 测试数据加载与分割
python features.py       # 测试特征抽出
python methods.py        # 测试5个手法都能跑通
python evaluator.py      # 测试评价函数
```

## 各模块说明

### config.py
所有可调参数集中在这里。修改任何参数都不需要动其他文件。

主要参数:
- `INTERPOLATION_METHOD`: 插补方法,可选 `'linear'` / `'ffill'` / `'kalman'`
- `BLOCK_SIZE_MIN`: 训练/测试块大小(分钟),默认5
- `WINDOW_SIZE`: 滑动窗口大小(秒),默认30
- `HPARAMS`: 各手法的默认超参数
- `HPARAM_GRID`: 超参数扫描范围(后续会用)

### data_loader.py
- `load_raw()`: 读取CSV log
- `resample_and_interpolate()`: 等间隔化(1秒)+ 插补
- `assign_blocks()`: 5分钟分块,奇偶分配train/test
- `assign_anomaly_types()`: 按时间标记5种异常类型
- `prepare_data()`: 一站式调用以上所有

### features.py
- `extract_window_features()`: 对每个窗口抽出7个统计特征
- `prepare_train_test()`: 输出训练/测试用的(X_train, y_train, X_test, y_test)

特征包括: mean, std, min, max, range, last, slope

### methods.py
统一接口的5个手法:
- `BaselineThreshold` (k = mu - k*sigma)
- `LogRegMethod`
- `RandomForestMethod`
- `IsolationForestMethod`
- `OneClassSVMMethod`

每个都有 `fit(X_train, y_train)` 和 `predict(X_test)` 方法。

### evaluator.py
- `evaluate(y_true, y_pred)`: 返回 {detection_rate, false_alarm_rate, TP, FN, FP, TN}
- `evaluate_by_anomaly_type()`: 按异常类型breakdown的评价

### main_pipeline.py
完整pipeline: 数据加载 → 前处理 → 特征 → 训练 → 评价
返回所有手法的整体性能 + 异常类型breakdown。

## 自定义实验示例

### 修改超参数
编辑 `config.py`,例如想改窗口大小:
```python
WINDOW_SIZE = 60  # 改成60秒
```
然后重新运行 `python main_pipeline.py`。

### 切换插补方法
```python
INTERPOLATION_METHOD = 'kalman'  # 改成卡尔曼滤波
```

### 在代码中调用
```python
from main_pipeline import run_pipeline
output = run_pipeline(verbose=True)

# output['results']: 各手法的整体评价
# output['results_by_type']: 按异常类型的breakdown
# output['data']: 训练/测试数据
# output['predictions']: 各手法的预测结果
```

## 数据集说明

- 测定日: 2025/12/19 14:30~19:30 (5小时)
- BLE传感器1台,放入金属储物柜
- 接收机距离约6m
- 异常 = 关闭储物柜门 → 金属遮蔽导致RSSI衰减

测定时段:
| 时段(分钟) | 模式 |
|---|---|
| 0–150 | 全部正常 |
| 150–180 | 15秒关闭(短期异常) |
| 180–210 | 30秒关闭(中期异常) |
| 210–240 | 60秒关闭(长期异常) |
| 240–270 | 缓慢关闭+保持(缓慢衰减) |
| 270–300 | 正常异常交替 |
