import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# 设置文件路径
base_path = os.path.dirname(__file__)
train_path = os.path.join(base_path, 'train_features.csv')
test_path = os.path.join(base_path, 'test_features.csv')
output_path = os.path.join(base_path, 'time_based_predictions.csv')
model_path = os.path.join(base_path, 'time_based_model.json')
feature_importance_plot_path = os.path.join(base_path, 'feature_importance_plot.png')

# 读取按时间分割的训练集和测试集
def load_time_based_data():
    print("开始加载时间分割的训练测试集...")
    
    # 读取训练集
    df_train = pd.read_csv(train_path)
    print(f"训练集加载完成，形状：{df_train.shape}")
    print(f"训练集目标分布：{df_train['target_high_value'].value_counts().to_dict()}")
    
    # 读取测试集
    df_test = pd.read_csv(test_path)
    print(f"测试集加载完成，形状：{df_test.shape}")
    print(f"测试集目标分布：{df_test['target_high_value'].value_counts().to_dict()}")
    
    return df_train, df_test

# 数据准备
def prepare_data(df_train, df_test):
    print("\n开始数据准备...")
    
    # 分离特征和目标变量
    X_train = df_train.drop(['target_high_value', 'customer_id', 'stat_month'], axis=1)
    y_train = df_train['target_high_value']
    
    X_test = df_test.drop(['target_high_value', 'customer_id', 'stat_month'], axis=1)
    y_test = df_test['target_high_value']
    
    # 获取客户ID和统计月份用于结果输出
    customer_ids_test = df_test['customer_id']
    stat_months_test = df_test['stat_month']
    
    print(f"训练集特征数量：{X_train.shape[1]}")
    print(f"测试集特征数量：{X_test.shape[1]}")
    
    return X_train, X_test, y_train, y_test, customer_ids_test, stat_months_test

# 训练模型
def train_model(X_train, y_train):
    print("\n开始训练XGBoost模型...")
    
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
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=True
    )
    
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
    
    return y_pred_proba, y_pred

# 特征重要性分析
def analyze_feature_importance(model, X_train):
    print("\n特征重要性分析：")
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    
    # 排序并输出前20个重要特征
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print(importance_df.head(20))
    
    # 特征重要性可视化
    print("\n生成特征重要性可视化图表...")
    plt.figure(figsize=(12, 8))
    
    # 选择前20个重要特征
    top_features = importance_df.head(20)
    
    # 绘制水平条形图
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性分数')
    plt.ylabel('特征名称')
    plt.title('XGBoost模型特征重要性（前20名）')
    plt.gca().invert_yaxis()  # 重要性高的特征在顶部
    
    # 添加数值标签
    for i, v in enumerate(top_features['importance']):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(feature_importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存到：{feature_importance_plot_path}")
    
    return importance_df

# 生成预测结果
def generate_predictions(y_test, y_pred_proba, y_pred, customer_ids_test, stat_months_test):
    print("\n开始生成预测结果...")
    
    # 创建预测结果数据框
    predictions = pd.DataFrame({
        'customer_id': customer_ids_test,
        'stat_month': stat_months_test,
        'actual_high_value': y_test,
        'predicted_high_value': y_pred,
        'prediction_probability': y_pred_proba
    })
    
    # 保存预测结果
    predictions.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到：{output_path}")
    
    # 输出高价值客户预测情况
    high_value_predictions = predictions[predictions['prediction_probability'] >= 0.7]
    print(f"\n高价值客户预测情况：")
    print(f"- 预测概率≥0.7的客户数：{len(high_value_predictions)}")
    print(f"- 其中实际为高价值客户的比例：{high_value_predictions['actual_high_value'].mean() * 100:.2f}%")
    
    return predictions

# 主函数
def main():
    print("=== 时间序列分类建模 - 高价值客户预测 ===")
    
    # 加载按时间分割的数据
    df_train, df_test = load_time_based_data()
    
    # 数据准备
    X_train, X_test, y_train, y_test, customer_ids_test, stat_months_test = prepare_data(df_train, df_test)
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # 评估模型
    y_pred_proba, y_pred = evaluate_model(model, X_test, y_test)
    
    # 特征重要性分析
    feature_importance = analyze_feature_importance(model, X_train)
    
    # 生成预测结果
    predictions = generate_predictions(y_test, y_pred_proba, y_pred, customer_ids_test, stat_months_test)
    
    print("\n=== 建模完成 ===")
    print(f"模型保存路径：{model_path}")
    print(f"预测结果保存路径：{output_path}")
    print(f"特征重要性图路径：{feature_importance_plot_path}")
    
    # 输出预测概率最高的前10个客户
    top_predictions = predictions.sort_values('prediction_probability', ascending=False).head(10)
    print("\n预测概率最高的前10个客户：")
    print(top_predictions[['customer_id', 'stat_month', 'actual_high_value', 'prediction_probability']])

if __name__ == "__main__":
    main()