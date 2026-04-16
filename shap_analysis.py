import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
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

# SHAP全局解释分析
def shap_global_explanation(gbm, X_train, X_test, feature_cols):
    print("\n进行SHAP全局解释分析...")
    
    # 初始化SHAP解释器
    explainer = shap.Explainer(gbm)
    
    # 计算SHAP值
    shap_values = explainer(X_test)
    
    # 1. 生成SHAP摘要图（全局特征重要性）
    print("\n生成SHAP摘要图...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, plot_type="bar", show=False)
    plt.title("SHAP全局特征重要性（按SHAP值绝对值平均）")
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'shap_global_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP全局特征重要性图已保存到: shap_global_importance.png")
    
    # 2. 生成SHAP蜂群图（全局特征重要性+特征与目标关系）
    print("\n生成SHAP蜂群图...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title("SHAP全局特征重要性与特征-目标关系")
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'shap_beeswarm_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP蜂群图已保存到: shap_beeswarm_plot.png")
    
    # 3. 计算并保存SHAP特征重要性数据
    shap_importance = np.abs(shap_values.values).mean(0)
    shap_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': shap_importance
    })
    shap_importance_df = shap_importance_df.sort_values(by='shap_importance', ascending=False)
    shap_importance_df.to_csv(os.path.join(base_path, 'shap_feature_importance.csv'), index=False, encoding='utf-8')
    print(f"SHAP特征重要性数据已保存到: shap_feature_importance.csv")
    
    # 打印SHAP特征重要性
    print("\nSHAP特征重要性排序:")
    print("特征名称			SHAP重要性分数")
    print("-" * 50)
    for _, row in shap_importance_df.iterrows():
        print(f"{row['feature']:<30} {row['shap_importance']:.6f}")
    
    return shap_values, shap_importance_df

# SHAP局部解释分析
def shap_local_explanation(gbm, explainer, shap_values, X_test, y_test, y_pred, feature_cols, df_test):
    print("\n进行SHAP局部解释分析...")
    
    # 1. 选择典型客户进行分析
    # 选择预测为高价值的客户
    high_value_indices = [i for i, (pred, true) in enumerate(zip(y_pred, y_test)) if pred > 0.5]
    # 选择预测为非高价值的客户
    low_value_indices = [i for i, (pred, true) in enumerate(zip(y_pred, y_test)) if pred <= 0.5]
    
    # 确保有足够的客户进行分析
    if len(high_value_indices) >= 2 and len(low_value_indices) >= 2:
        # 选择2个高价值客户和2个非高价值客户
        sample_indices = high_value_indices[:2] + low_value_indices[:2]
    else:
        # 如果数量不足，选择所有可用的客户
        sample_indices = high_value_indices + low_value_indices
    
    # 2. 为每个选定客户生成SHAP局部解释
    for i, idx in enumerate(sample_indices):
        print(f"\n生成客户 {i+1} 的SHAP局部解释...")
        
        # 获取客户ID
        customer_id = df_test.iloc[idx]['customer_id']
        
        # 生成SHAP瀑布图（单个客户解释）
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
        plt.title(f"客户 {customer_id} 的SHAP局部解释")
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'shap_local_explanation_customer_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"客户 {customer_id} 的SHAP局部解释图已保存到: shap_local_explanation_customer_{i+1}.png")
        
        # 生成SHAP力图（单个客户解释）
        plt.figure(figsize=(12, 8))
        shap.plots.force(shap_values[idx], show=False)
        plt.title(f"客户 {customer_id} 的SHAP力图")
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'shap_force_plot_customer_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"客户 {customer_id} 的SHAP力图已保存到: shap_force_plot_customer_{i+1}.png")
    
    # 3. 保存客户SHAP值分析数据
    customer_shap_df = pd.DataFrame({
        'customer_id': df_test.iloc[sample_indices]['customer_id'].values,
        'predicted_probability': y_pred[sample_indices],
        'actual_target': y_test.iloc[sample_indices].values
    })
    
    # 添加每个特征的SHAP值
    for i, feature in enumerate(feature_cols):
        customer_shap_df[f'shap_{feature}'] = shap_values.values[sample_indices, i]
    
    customer_shap_df.to_csv(os.path.join(base_path, 'shap_customer_analysis.csv'), index=False, encoding='utf-8')
    print(f"客户SHAP值分析数据已保存到: shap_customer_analysis.csv")
    
    return customer_shap_df

# 主函数
def main():
    print("=== 潜在高价值客户预测 - SHAP分析 ===")
    
    # 加载数据
    df_train, df_test = load_data()
    
    # 准备特征和目标变量
    X_train, X_test, y_train, y_test, feature_cols = prepare_features(df_train, df_test)
    
    # 训练LightGBM模型
    gbm = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # 评估模型
    y_pred, y_pred_binary = evaluate_model(gbm, X_test, y_test)
    
    # 初始化SHAP解释器
    explainer = shap.Explainer(gbm)
    
    # 计算SHAP值
    print("\n计算SHAP值...")
    shap_values = explainer(X_test)
    
    # SHAP全局解释分析
    shap_global_explanation(gbm, X_train, X_test, feature_cols)
    
    # SHAP局部解释分析
    shap_local_explanation(gbm, explainer, shap_values, X_test, y_test, y_pred, feature_cols, df_test)
    
    print("\n=== SHAP分析完成 ===")
    print(f"生成的文件:")
    print(f"- SHAP全局特征重要性图: {os.path.join(base_path, 'shap_global_importance.png')}")
    print(f"- SHAP蜂群图: {os.path.join(base_path, 'shap_beeswarm_plot.png')}")
    print(f"- SHAP特征重要性数据: {os.path.join(base_path, 'shap_feature_importance.csv')}")
    print(f"- 客户SHAP值分析数据: {os.path.join(base_path, 'shap_customer_analysis.csv')}")
    print(f"- 客户SHAP局部解释图: shap_local_explanation_customer_*.png")
    print(f"- 客户SHAP力图: shap_force_plot_customer_*.png")

if __name__ == "__main__":
    main()