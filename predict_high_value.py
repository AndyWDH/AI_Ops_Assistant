import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 设置文件路径
base_path = os.path.dirname(__file__)
features_path = os.path.join(base_path, 'processed_features.csv')
output_path = os.path.join(base_path, 'high_value_predictions.csv')
model_path = os.path.join(base_path, 'high_value_model.json')

# 读取处理后的特征数据
def load_data(sample_size=10000):
    print("开始加载数据...")
    df = pd.read_csv(features_path)
    print(f"数据加载完成，原始形状：{df.shape}")
    print(f"目标变量分布：{df['target_high_value'].value_counts().to_dict()}")
    
    # 使用采样数据进行测试，加快运行速度
    if sample_size and sample_size < len(df):
        # 移除stratify参数，因为df.sample()不支持
        df = df.sample(n=sample_size, random_state=42)
        print(f"使用采样数据，采样后形状：{df.shape}")
        print(f"采样后目标变量分布：{df['target_high_value'].value_counts().to_dict()}")
    
    return df

# 数据准备
def prepare_data(df):
    print("\n开始数据准备...")
    
    # 分离特征和目标变量
    X = df.drop(['target_high_value', 'customer_id', 'stat_month'], axis=1)
    y = df['target_high_value']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, customer_ids_train, customer_ids_test = train_test_split(
        X, y, df['customer_id'], test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集形状：{X_train.shape}")
    print(f"测试集形状：{X_test.shape}")
    print(f"训练集目标分布：{y_train.value_counts().to_dict()}")
    print(f"测试集目标分布：{y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, customer_ids_test

# 训练模型
def train_model(X_train, y_train):
    print("\n开始训练模型...")
    
    # 初始化XGBoost分类器
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"交叉验证AUC分数：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 保存模型
    model.save_model(model_path)
    print(f"模型已保存到：{model_path}")
    
    return model

# 评估模型
def evaluate_model(model, X_test, y_test):
    print("\n开始评估模型...")
    
    # 预测概率
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 预测类别
    y_pred = model.predict(X_test)
    
    # 计算AUC分数
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"测试集AUC分数：{roc_auc:.4f}")
    
    # 计算PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"测试集PR AUC分数：{pr_auc:.4f}")
    
    # 分类报告
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵：")
    print(cm)
    
    return y_pred_proba

# 特征重要性分析
def analyze_feature_importance(model, X_train):
    print("\n特征重要性分析：")
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    
    # 排序并输出前20个重要特征
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print(importance_df.head(20))
    
    return importance_df

# 筛选高价值客户
def filter_high_value_customers(X_test, y_test, y_pred_proba, customer_ids_test, threshold=0.7):
    print(f"\n开始筛选高价值客户，阈值：{threshold}")
    
    # 创建测试结果数据框
    test_results = pd.DataFrame({
        'customer_id': customer_ids_test,
        'target_high_value': y_test.values,
        'prediction_prob': y_pred_proba
    })
    
    # 筛选高概率用户
    high_value_customers = test_results[test_results['prediction_prob'] >= threshold]
    high_value_customers = high_value_customers.sort_values('prediction_prob', ascending=False)
    
    print(f"筛选出的高价值客户数量：{len(high_value_customers)}")
    print(f"高价值客户占测试集比例：{len(high_value_customers) / len(test_results) * 100:.2f}%")
    print(f"高价值客户中实际为高价值的比例：{high_value_customers['target_high_value'].mean() * 100:.2f}%")
    
    # 保存结果
    high_value_customers.to_csv(output_path, index=False, encoding='utf-8')
    print(f"高价值客户列表已保存到：{output_path}")
    
    return high_value_customers

# 主函数
def main():
    # 加载数据
    df = load_data()
    
    # 数据准备
    X_train, X_test, y_train, y_test, customer_ids_test = prepare_data(df)
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # 评估模型
    y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 特征重要性分析
    feature_importance = analyze_feature_importance(model, X_train)
    
    # 筛选高价值客户
    high_value_customers = filter_high_value_customers(X_test, y_test, y_pred_proba, customer_ids_test)
    
    print("\n建模完成！")
    print(f"\n高价值客户预测结果：")
    print(f"- 预测概率最高的前10个客户：")
    print(high_value_customers[['customer_id', 'prediction_prob', 'target_high_value']].head(10))

if __name__ == "__main__":
    main()