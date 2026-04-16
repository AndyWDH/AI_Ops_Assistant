import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 设置文件路径
base_path = os.path.dirname(__file__)
train_path = os.path.join(base_path, 'train_features.csv')
test_path = os.path.join(base_path, 'test_features.csv')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
def load_data():
    print("加载数据...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 显示数据基本信息
    print(f"训练集形状: {df_train.shape}")
    print(f"测试集形状: {df_test.shape}")
    print(f"训练集高价值客户占比: {df_train['target_high_value'].mean():.2%}")
    print(f"测试集高价值客户占比: {df_test['target_high_value'].mean():.2%}")
    
    return df_train, df_test

# 准备特征和目标变量
def prepare_features(df_train, df_test):
    print("\n准备特征和目标变量...")
    
    # 定义特征列（排除目标变量、customer_id和stat_month）
    feature_cols = [col for col in df_train.columns 
                   if col not in ['target_high_value', 'customer_id', 'stat_month']]
    
    # 准备训练数据
    X_train = df_train[feature_cols]
    y_train = df_train['target_high_value']
    
    # 准备测试数据
    X_test = df_test[feature_cols]
    y_test = df_test['target_high_value']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")
    
    return X_train, X_test, y_train, y_test, feature_cols

# 训练LightGBM模型
def train_lightgbm(X_train, y_train, X_test, y_test):
    print(f"\n训练LightGBM模型...")
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 设置模型参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42,
        'max_depth': 4,
        'n_jobs': -1
    }
    
    # 训练模型
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    print("模型训练完成")
    return gbm

# 模型评估
def evaluate_model(gbm, X_test, y_test):
    print("\n评估模型性能...")
    
    # 预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_binary))
    
    return y_pred, y_pred_binary

# 输出特征重要性（文本打印）
def print_feature_importance(gbm, feature_cols):
    print("\n特征重要性排序（文本打印）:")
    
    # 获取特征重要性
    importance = gbm.feature_importance(importance_type='split')
    feature_importance = list(zip(feature_cols, importance))
    
    # 按重要性排序
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # 打印特征重要性
    print("特征名称			重要性分数")
    print("-" * 50)
    for feature, score in feature_importance:
        print(f"{feature:<30} {score:.6f}")
    
    # 保存特征重要性到文件
    importance_df = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    importance_df.to_csv(os.path.join(base_path, 'lightgbm_feature_importance.csv'), index=False, encoding='utf-8')
    print(f"\n特征重要性已保存到: lightgbm_feature_importance.csv")
    
    return importance_df

# 生成特征重要性可视化图片
def plot_feature_importance(importance_df):
    print("\n生成特征重要性可视化图片...")
    
    # 取前20个重要特征
    top_features = importance_df.head(20)
    
    # 创建可视化
    plt.figure(figsize=(12, 8))
    
    # 绘制水平条形图
    plt.barh(range(len(top_features)), top_features['importance'], color='#3498db')
    
    # 设置y轴标签
    plt.yticks(range(len(top_features)), top_features['feature'])
    
    # 设置x轴标签
    plt.xlabel('特征重要性分数')
    
    # 设置标题
    plt.title('LightGBM模型特征重要性排序（前20名）')
    
    # 反转y轴，使重要性高的特征在顶部
    plt.gca().invert_yaxis()
    
    # 添加数值标签
    for i, v in enumerate(top_features['importance']):
        plt.text(v + 0.5, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(base_path, 'lightgbm_feature_importance_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"特征重要性图已保存到: {output_path}")
    return output_path

# 主函数
def main():
    print("=== 潜在高价值客户预测 - LightGBM分析 ===")
    
    # 加载数据
    df_train, df_test = load_data()
    
    # 准备特征和目标变量
    X_train, X_test, y_train, y_test, feature_cols = prepare_features(df_train, df_test)
    
    # 训练LightGBM模型
    gbm = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # 评估模型
    y_pred, y_pred_binary = evaluate_model(gbm, X_test, y_test)
    
    # 输出特征重要性（文本打印）
    importance_df = print_feature_importance(gbm, feature_cols)
    
    # 生成特征重要性可视化图片
    plot_path = plot_feature_importance(importance_df)
    
    print("\n=== 分析完成 ===")
    print(f"生成的文件:")
    print(f"- 特征重要性数据: {os.path.join(base_path, 'lightgbm_feature_importance.csv')}")
    print(f"- 特征重要性可视化: {plot_path}")

if __name__ == "__main__":
    main()
