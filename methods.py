"""
methods.py
5个异常检测手法的统一封装
所有手法都实装为类,提供 fit() 和 predict() 接口
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

import config


# ==========================================
# Baseline: 统计阈值法
# ==========================================
class BaselineThreshold:
    """
    基于训练集正常数据的mean、std,用 mu - k*sigma 作为阈值
    判定: 窗口的mean < threshold → 异常
    """
    name = 'Baseline'
    
    def __init__(self, k=1.5):
        self.k = k
        self.threshold = None
    
    def fit(self, X_train, y_train):
        # 用训练集中label==0的窗口的mean
        # X_train的第0列是'mean'特征
        normal_means = X_train[y_train == 0, 0]
        mu = normal_means.mean()
        sigma = normal_means.std()
        self.threshold = mu - self.k * sigma
        return self
    
    def predict(self, X_test):
        return (X_test[:, 0] < self.threshold).astype(int)


# ==========================================
# 监督学习①: Logistic Regression
# ==========================================
class LogRegMethod:
    name = 'LogReg'
    
    def __init__(self, C=1.0, class_weight='balanced', max_iter=1000):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.scaler = None
        self.model = None
    
    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
        )
        self.model.fit(X_train_s, y_train)
        return self
    
    def predict(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        return self.model.predict(X_test_s)


# ==========================================
# 监督学习②: Random Forest
# ==========================================
class RandomForestMethod:
    name = 'RandomForest'
    
    def __init__(self, n_estimators=100, max_depth=None,
                 class_weight='balanced', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None
    
    def fit(self, X_train, y_train):
        # RF不需要标准化
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        return self.model.predict(X_test)


# ==========================================
# 无监督学习③: Isolation Forest
# ==========================================
class IsolationForestMethod:
    name = 'IsolationForest'
    
    def __init__(self, n_estimators=100, contamination=0.07, random_state=42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = None
        self.model = None
    
    def fit(self, X_train, y_train=None):
        # 无监督: 不使用y_train (但接口保持一致)
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(X_train_s)
        return self
    
    def predict(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        # IForest: -1=异常, 1=正常 → 转换为 1=异常, 0=正常
        return (self.model.predict(X_test_s) == -1).astype(int)


# ==========================================
# 无监督学习④: One-Class SVM
# ==========================================
class OneClassSVMMethod:
    name = 'OneClassSVM'
    
    def __init__(self, nu=0.07, kernel='rbf', gamma='scale'):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = None
        self.model = None
    
    def fit(self, X_train, y_train=None):
        # OC-SVM: 仅用正常数据训练
        # y_train==None 时,假定全部都是正常
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        if y_train is not None:
            X_train_normal = X_train_s[y_train == 0]
        else:
            X_train_normal = X_train_s
        
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
        )
        self.model.fit(X_train_normal)
        return self
    
    def predict(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        return (self.model.predict(X_test_s) == -1).astype(int)


# ==========================================
# 工厂函数: 用默认参数构造各手法
# ==========================================
def get_default_methods():
    """返回5个手法的实例列表 (按发表顺序)"""
    return [
        BaselineThreshold(**config.HPARAMS['Baseline']),
        LogRegMethod(**config.HPARAMS['LogisticRegression']),
        RandomForestMethod(**config.HPARAMS['RandomForest']),
        IsolationForestMethod(**config.HPARAMS['IsolationForest']),
        OneClassSVMMethod(**config.HPARAMS['OneClassSVM']),
    ]


if __name__ == '__main__':
    # 测试每个手法都能跑通
    import data_loader, features
    df = data_loader.prepare_data()
    data = features.prepare_train_test(df)
    
    print("测试各手法的 fit/predict ...")
    for method in get_default_methods():
        method.fit(data['X_train'], data['y_train'])
        y_pred = method.predict(data['X_test'])
        print(f"  {method.name}: 输出形状 {y_pred.shape}, 异常预测数 {y_pred.sum()}")
    print("OK!")
