import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# 设置文件路径
base_path = os.path.dirname(__file__)
train_path = os.path.join(base_path, 'train_features.csv')
test_path = os.path.join(base_path, 'test_features.csv')

# 加载数据
def load_data():
    print("加载数据...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

# 准备特征和目标变量
def prepare_features(df_train, df_test):
    print("准备特征和目标变量...")
    feature_cols = [col for col in df_train.columns 
                   if col not in ['target_high_value', 'customer_id', 'stat_month']]
    
    X_train = df_train[feature_cols]
    y_train = df_train['target_high_value']
    
    X_test = df_test[feature_cols]
    y_test = df_test['target_high_value']
    
    return X_train, X_test, y_train, y_test, feature_cols

# 训练LightGBM模型
def train_lightgbm(X_train, y_train, X_test, y_test):
    print("训练LightGBM模型...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
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
    
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    return gbm

# 模型评估
def evaluate_model(gbm, X_test, y_test):
    print("评估模型性能...")
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return y_pred, y_pred_binary

# 全局解释 - 特征重要性
def global_explanation(gbm, feature_cols):
    print("\n=== 全局解释 - 特征重要性 ===")
    
    # 获取特征重要性
    importance_split = gbm.feature_importance(importance_type='split')
    importance_gain = gbm.feature_importance(importance_type='gain')
    
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_split': importance_split,
        'importance_gain': importance_gain
    })
    
    # 按重要性排序（使用gain作为主要排序依据）
    importance_df = importance_df.sort_values('importance_gain', ascending=False)
    
    # 打印特征重要性排序
    print("\n特征重要性排序（基于gain）:")
    print("特征名称\t\t\t重要性(gain)")
    print("-" * 50)
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:<30} {row['importance_gain']:<15.2f}")
    
    # 保存特征重要性到文件
    importance_df.to_csv(os.path.join(base_path, 'simple_feature_importance.csv'), index=False, encoding='utf-8')
    print(f"\n特征重要性已保存到: simple_feature_importance.csv")
    
    return importance_df

# 局部解释 - 基于特征贡献
def local_explanation(gbm, X_test, feature_cols, y_pred, y_pred_binary):
    print("\n=== 局部解释 - 特征贡献分析 ===")
    
    # 选择不同预测结果的客户
    high_value_indices = [i for i, binary_pred in enumerate(y_pred_binary) if binary_pred == 1][:2]
    low_value_indices = [i for i, binary_pred in enumerate(y_pred_binary) if binary_pred == 0][:2]
    selected_indices = high_value_indices + low_value_indices
    
    # 获取模型的基础分数
    base_score = gbm.predict([X_test.iloc[0].tolist()], num_iteration=gbm.best_iteration, pred_contrib=True)[0][-1]
    print(f"模型基础分数: {base_score:.4f}")
    
    for idx in selected_indices:
        print(f"\n客户 {idx+1} 预测概率: {y_pred[idx]:.4f}, 预测类别: {y_pred_binary[idx]}")
        
        # 获取特征贡献
        contrib = gbm.predict([X_test.iloc[idx].tolist()], num_iteration=gbm.best_iteration, pred_contrib=True)[0]
        feature_contrib = contrib[:-1]
        
        # 创建特征贡献数据框
        contrib_df = pd.DataFrame({
            'feature': feature_cols,
            'value': X_test.iloc[idx].tolist(),
            'contribution': feature_contrib
        })
        
        # 按贡献绝对值排序
        contrib_df = contrib_df.sort_values('contribution', key=lambda x: abs(x), ascending=False)
        
        # 打印前10个贡献最大的特征
        print("前10个贡献最大的特征:")
        print("特征名称\t\t\t特征值\t\t贡献值")
        print("-" * 70)
        for _, row in contrib_df.head(10).iterrows():
            print(f"{row['feature']:<30} {row['value']:<15.4f} {row['contribution']:<15.4f}")
        
        # 计算正负贡献总和
        total_positive = contrib_df[contrib_df['contribution'] > 0]['contribution'].sum()
        total_negative = contrib_df[contrib_df['contribution'] < 0]['contribution'].sum()
        print(f"\n总正贡献: {total_positive:.4f}, 总负贡献: {total_negative:.4f}")
        
        # 保存特征贡献到文件
        contrib_df.to_csv(os.path.join(base_path, f'simple_customer_{idx+1}_contribution.csv'), index=False, encoding='utf-8')

# 主函数
def main():
    print("=== 潜在高价值客户预测 - 简化版模型解释 ===")
    
    # 加载数据
    df_train, df_test = load_data()
    
    # 准备特征和目标变量
    X_train, X_test, y_train, y_test, feature_cols = prepare_features(df_train, df_test)
    
    # 训练LightGBM模型
    gbm = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # 评估模型
    y_pred, y_pred_binary = evaluate_model(gbm, X_test, y_test)
    
    # 全局解释
    importance_df = global_explanation(gbm, feature_cols)
    
    # 局部解释
    local_explanation(gbm, X_test, feature_cols, y_pred, y_pred_binary)
    
    print("\n=== 模型解释完成 ===")
    print("生成的文件:")
    print("- simple_feature_importance.csv: 特征重要性排序")
    print("- simple_customer_*_contribution.csv: 客户特征贡献")

if __name__ == "__main__":
    main()
