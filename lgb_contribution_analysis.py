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

# 全局特征重要性分析（类似SHAP全局解释）
def global_feature_importance(gbm, X_train, X_test, feature_cols):
    print("\n进行全局特征重要性分析...")
    
    # 1. 获取不同类型的特征重要性
    # 基于分裂次数的特征重要性
    importance_split = gbm.feature_importance(importance_type='split')
    # 基于增益的特征重要性
    importance_gain = gbm.feature_importance(importance_type='gain')
    
    # 2. 计算特征贡献（类似SHAP值）
    print("计算特征贡献值...")
    feature_contrib = gbm.predict(X_test, num_iteration=gbm.best_iteration, pred_contrib=True)
    
    # 排除偏置项（最后一列）
    feature_contrib = feature_contrib[:, :-1]
    
    # 计算平均绝对贡献（类似SHAP全局重要性）
    avg_abs_contrib = np.abs(feature_contrib).mean(axis=0)
    
    # 3. 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_split': importance_split,
        'importance_gain': importance_gain,
        'avg_abs_contrib': avg_abs_contrib
    })
    
    # 按平均绝对贡献排序（类似SHAP重要性）
    importance_df = importance_df.sort_values(by='avg_abs_contrib', ascending=False)
    
    # 4. 保存特征重要性数据
    importance_df.to_csv(os.path.join(base_path, 'lgb_global_importance.csv'), index=False, encoding='utf-8')
    print(f"全局特征重要性数据已保存到: lgb_global_importance.csv")
    
    # 5. 打印特征重要性
    print("\n全局特征重要性排序（按平均绝对贡献）:")
    print("特征名称			平均绝对贡献")
    print("-" * 50)
    for _, row in importance_df.head(20).iterrows():
        print(f"{row['feature']:<30} {row['avg_abs_contrib']:.6f}")
    
    # 6. 生成全局特征重要性可视化
    print("\n生成全局特征重要性可视化...")
    
    # 绘制前20个重要特征
    top_features = importance_df.head(20)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['avg_abs_contrib'], color='#3498db')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('平均绝对贡献（类似SHAP值）')
    plt.title('全局特征重要性（按平均绝对贡献排序）')
    plt.gca().invert_yaxis()
    
    # 添加数值标签
    for i, v in enumerate(top_features['avg_abs_contrib']):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'lgb_global_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"全局特征重要性图已保存到: lgb_global_importance.png")
    
    return importance_df, feature_contrib

# 局部特征贡献分析（类似SHAP局部解释）
def local_feature_contribution(gbm, X_test, y_test, y_pred, feature_cols, df_test, feature_contrib):
    print("\n进行局部特征贡献分析...")
    
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
    
    # 2. 为每个选定客户生成局部解释
    for i, idx in enumerate(sample_indices):
        print(f"\n生成客户 {i+1} 的局部特征贡献解释...")
        
        # 获取客户ID
        customer_id = df_test.iloc[idx]['customer_id']
        
        # 获取该客户的特征贡献
        contrib = feature_contrib[idx]
        
        # 创建局部贡献数据框
        local_contrib_df = pd.DataFrame({
            'feature': feature_cols,
            'contribution': contrib
        })
        
        # 按贡献绝对值排序
        local_contrib_df = local_contrib_df.sort_values(by='contribution', ascending=False)
        
        # 3. 保存客户局部贡献数据
        local_contrib_df.to_csv(os.path.join(base_path, f'lgb_local_contrib_customer_{i+1}.csv'), index=False, encoding='utf-8')
        
        # 4. 生成局部贡献可视化（类似SHAP瀑布图）
        plt.figure(figsize=(12, 8))
        
        # 选择前15个最重要的特征（按贡献绝对值）
        top_local_contrib = local_contrib_df.sort_values(by='contribution', key=abs, ascending=False).head(15)
        
        # 按贡献值排序（正数在上，负数在下）
        top_local_contrib = top_local_contrib.sort_values(by='contribution', ascending=True)
        
        # 绘制水平条形图
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_local_contrib['contribution']]
        plt.barh(range(len(top_local_contrib)), top_local_contrib['contribution'], color=colors)
        plt.yticks(range(len(top_local_contrib)), top_local_contrib['feature'])
        plt.xlabel('特征贡献值')
        plt.title(f'客户 {customer_id} 的特征贡献分析（类似SHAP局部解释）')
        
        # 添加数值标签
        for j, v in enumerate(top_local_contrib['contribution']):
            if v > 0:
                plt.text(v + 0.001, j, f'{v:.4f}', va='center')
            else:
                plt.text(v - 0.02, j, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'lgb_local_contrib_customer_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"客户 {customer_id} 的局部贡献图已保存到: lgb_local_contrib_customer_{i+1}.png")
    
    # 5. 保存所有客户的局部贡献数据
    all_contrib_df = pd.DataFrame(feature_contrib, columns=feature_cols)
    all_contrib_df['customer_id'] = df_test['customer_id'].values
    all_contrib_df['predicted_probability'] = y_pred
    all_contrib_df['actual_target'] = y_test.values
    all_contrib_df.to_csv(os.path.join(base_path, 'lgb_all_customer_contrib.csv'), index=False, encoding='utf-8')
    print(f"所有客户的局部贡献数据已保存到: lgb_all_customer_contrib.csv")
    
    return all_contrib_df

# 主函数
def main():
    print("=== 潜在高价值客户预测 - LightGBM特征贡献分析（类似SHAP）===")
    
    # 加载数据
    df_train, df_test = load_data()
    
    # 准备特征和目标变量
    X_train, X_test, y_train, y_test, feature_cols = prepare_features(df_train, df_test)
    
    # 训练LightGBM模型
    gbm = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # 评估模型
    y_pred, y_pred_binary = evaluate_model(gbm, X_test, y_test)
    
    # 全局特征重要性分析（类似SHAP全局解释）
    importance_df, feature_contrib = global_feature_importance(gbm, X_train, X_test, feature_cols)
    
    # 局部特征贡献分析（类似SHAP局部解释）
    all_contrib_df = local_feature_contribution(gbm, X_test, y_test, y_pred, feature_cols, df_test, feature_contrib)
    
    print("\n=== 特征贡献分析完成 ===")
    print(f"生成的文件:")
    print(f"- 全局特征重要性数据: {os.path.join(base_path, 'lgb_global_importance.csv')}")
    print(f"- 全局特征重要性图: {os.path.join(base_path, 'lgb_global_importance.png')}")
    print(f"- 所有客户的局部贡献数据: {os.path.join(base_path, 'lgb_all_customer_contrib.csv')}")
    print(f"- 单个客户的局部贡献数据: lgb_local_contrib_customer_*.csv")
    print(f"- 单个客户的局部贡献图: lgb_local_contrib_customer_*.png")

if __name__ == "__main__":
    main()