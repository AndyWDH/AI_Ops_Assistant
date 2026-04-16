import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

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
    
    return X_train, X_test, y_train, y_test, feature_cols, df_train, df_test

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

# 全局解释 - 特征重要性
def global_explanation(gbm, X_train, feature_cols):
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
    print("特征名称\t\t\t重要性(gain)\t重要性(split)")
    print("-" * 60)
    for _, row in importance_df.iterrows():
        print(f"{row['feature']:<30} {row['importance_gain']:<15.2f} {row['importance_split']:<15.2f}")
    
    # 保存特征重要性到文件
    importance_df.to_csv(os.path.join(base_path, 'feature_importance.csv'), index=False, encoding='utf-8')
    print(f"\n特征重要性已保存到: feature_importance.csv")
    
    # 可视化特征重要性
    visualize_feature_importance(importance_df)
    
    return importance_df

# 可视化特征重要性
def visualize_feature_importance(importance_df):
    print("\n生成特征重要性可视化...")
    
    # 取前20个重要特征
    top_features = importance_df.head(20)
    
    # 创建水平条形图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制gain重要性
    bars = ax.barh(range(len(top_features)), top_features['importance_gain'], color='#3498db', label='Importance (Gain)')
    
    # 设置y轴标签
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    
    # 设置x轴标签
    ax.set_xlabel('特征重要性分数')
    
    # 设置标题
    ax.set_title('LightGBM模型特征重要性排序（前20名，基于Gain）')
    
    # 反转y轴，使重要性高的特征在顶部
    ax.invert_yaxis()
    
    # 添加数值标签
    for i, v in enumerate(top_features['importance_gain']):
        ax.text(v + 0.5, i, f'{v:.2f}', va='center')
    
    ax.legend()
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(base_path, 'feature_importance_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存到: {output_path}")
    
    # 同时绘制两种重要性指标的对比图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置x轴位置
    x = np.arange(len(top_features))
    width = 0.35
    
    # 绘制两种重要性指标
    ax.barh(x - width/2, top_features['importance_gain'], width, label='Importance (Gain)')
    ax.barh(x + width/2, top_features['importance_split'], width, label='Importance (Split)')
    
    # 设置y轴标签
    ax.set_yticks(x)
    ax.set_yticklabels(top_features['feature'])
    
    # 设置x轴标签
    ax.set_xlabel('特征重要性分数')
    
    # 设置标题
    ax.set_title('LightGBM模型特征重要性对比（前20名）')
    
    # 反转y轴
    ax.invert_yaxis()
    
    ax.legend()
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(base_path, 'feature_importance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性对比图已保存到: {output_path}")

# 局部解释 - 基于特征贡献
def local_explanation(gbm, X_test, feature_cols, df_test, y_pred, y_pred_binary):
    print("\n=== 局部解释 - 特征贡献分析 ===")
    
    # 选择不同预测结果的客户进行局部解释
    # 1. 预测为高价值客户的客户
    high_value_indices = [i for i, (pred, binary_pred) in enumerate(zip(y_pred, y_pred_binary)) if binary_pred == 1][:3]
    
    # 2. 预测为非高价值客户的客户
    low_value_indices = [i for i, (pred, binary_pred) in enumerate(zip(y_pred, y_pred_binary)) if binary_pred == 0][:3]
    
    # 合并选择的客户
    selected_indices = high_value_indices + low_value_indices
    
    # 获取模型的基础分数（bias）
    base_score = gbm.predict([X_test.iloc[0].tolist()], num_iteration=gbm.best_iteration, pred_contrib=True)[0][-1]
    
    # 为每个选定客户生成局部解释
    for idx in selected_indices:
        print(f"\n为客户 {idx+1} 生成局部解释...")
        
        # 获取特征贡献
        contrib = gbm.predict([X_test.iloc[idx].tolist()], num_iteration=gbm.best_iteration, pred_contrib=True)[0]
        
        # 提取特征贡献（排除bias项）
        feature_contrib = contrib[:-1]
        
        # 创建特征贡献数据框
        contrib_df = pd.DataFrame({
            'feature': feature_cols,
            'value': X_test.iloc[idx].tolist(),
            'contribution': feature_contrib
        })
        
        # 计算标准化贡献（便于比较）
        total_contrib = np.abs(feature_contrib).sum()
        contrib_df['normalized_contribution'] = contrib_df['contribution'] / total_contrib if total_contrib != 0 else 0
        
        # 按贡献绝对值排序
        contrib_df = contrib_df.sort_values('contribution', key=lambda x: abs(x), ascending=False)
        
        # 打印前10个贡献最大的特征
        print(f"\n客户 {idx+1} 前10个贡献最大的特征:")
        print("特征名称\t\t\t特征值\t\t贡献值\t\t标准化贡献")
        print("-" * 80)
        for _, row in contrib_df.head(10).iterrows():
            print(f"{row['feature']:<30} {row['value']:<15.4f} {row['contribution']:<15.4f} {row['normalized_contribution']:<15.4f}")
        
        # 保存特征贡献到文件
        contrib_df.to_csv(os.path.join(base_path, f'customer_{idx+1}_contribution.csv'), index=False, encoding='utf-8')
        
        # 可视化特征贡献
        visualize_feature_contribution(contrib_df, idx+1, base_score, y_pred[idx])
        
        # 保存客户详细信息
        customer_info = df_test.iloc[idx].copy()
        customer_info['predicted_prob'] = y_pred[idx]
        customer_info['predicted_class'] = y_pred_binary[idx]
        customer_info['actual_class'] = df_test.iloc[idx]['target_high_value']
        customer_info['base_score'] = base_score
        customer_info.to_csv(os.path.join(base_path, f'customer_{idx+1}_info.csv'), encoding='utf-8')
        print(f"客户 {idx+1} 详细信息已保存到: customer_{idx+1}_info.csv")
    
    # 生成所有客户的贡献摘要
    generate_contribution_summary(gbm, X_test, feature_cols, y_pred)

# 可视化特征贡献
def visualize_feature_contribution(contrib_df, customer_id, base_score, predicted_prob):
    print(f"\n生成客户 {customer_id} 特征贡献可视化...")
    
    # 取前15个贡献最大的特征
    top_contrib = contrib_df.head(15)
    
    # 分类正负贡献
    positive_contrib = top_contrib[top_contrib['contribution'] > 0].sort_values('contribution', ascending=True)
    negative_contrib = top_contrib[top_contrib['contribution'] < 0].sort_values('contribution', ascending=True)
    
    # 创建瀑布图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 设置初始位置
    cumulative = base_score
    
    # 绘制基础分数
    ax.barh(0, base_score, color='#f0f0f0', label=f'基础分数: {base_score:.4f}')
    ax.text(base_score + 0.001, 0, f'基础分数: {base_score:.4f}', va='center')
    
    # 绘制负贡献
    y_pos = 1
    for _, row in negative_contrib.iterrows():
        ax.barh(y_pos, row['contribution'], color='#e74c3c', label='负贡献' if y_pos == 1 else "")
        ax.text(cumulative + row['contribution'] + 0.001, y_pos, f'{row["contribution"]:.4f}', va='center')
        ax.text(-0.1, y_pos, row['feature'], ha='right', va='center')
        cumulative += row['contribution']
        y_pos += 1
    
    # 绘制正贡献
    for _, row in positive_contrib.iterrows():
        ax.barh(y_pos, row['contribution'], color='#27ae60', label='正贡献' if y_pos == len(negative_contrib) + 1 else "")
        ax.text(cumulative + row['contribution'] + 0.001, y_pos, f'{row["contribution"]:.4f}', va='center')
        ax.text(-0.1, y_pos, row['feature'], ha='right', va='center')
        cumulative += row['contribution']
        y_pos += 1
    
    # 绘制最终预测分数
    ax.barh(y_pos, predicted_prob - cumulative, color='#3498db', label=f'最终预测概率: {predicted_prob:.4f}')
    ax.text(predicted_prob + 0.001, y_pos, f'最终预测: {predicted_prob:.4f}', va='center')
    ax.text(-0.1, y_pos, '最终预测', ha='right', va='center')
    
    # 设置标题
    ax.set_title(f'客户 {customer_id} 特征贡献瀑布图')
    
    # 设置x轴标签
    ax.set_xlabel('贡献值')
    
    # 隐藏y轴
    ax.get_yaxis().set_visible(False)
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(base_path, f'customer_{customer_id}_contribution_waterfall.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征贡献瀑布图已保存到: {output_path}")

# 生成贡献摘要
def generate_contribution_summary(gbm, X_test, feature_cols, y_pred):
    print("\n生成特征贡献摘要...")
    
    # 获取所有客户的特征贡献
    all_contrib = gbm.predict(X_test, num_iteration=gbm.best_iteration, pred_contrib=True)
    
    # 提取特征贡献（排除bias项）
    feature_contribs = all_contrib[:, :-1]
    
    # 计算平均贡献
    avg_contrib = feature_contribs.mean(axis=0)
    
    # 计算贡献的标准差
    std_contrib = feature_contribs.std(axis=0)
    
    # 计算正贡献客户比例
    positive_contrib_ratio = (feature_contribs > 0).mean(axis=0)
    
    # 创建贡献摘要数据框
    contrib_summary = pd.DataFrame({
        'feature': feature_cols,
        'average_contribution': avg_contrib,
        'std_contribution': std_contrib,
        'positive_contrib_ratio': positive_contrib_ratio
    })
    
    # 按平均贡献绝对值排序
    contrib_summary = contrib_summary.sort_values('average_contribution', key=lambda x: abs(x), ascending=False)
    
    # 保存贡献摘要到文件
    contrib_summary.to_csv(os.path.join(base_path, 'contribution_summary.csv'), index=False, encoding='utf-8')
    print(f"特征贡献摘要已保存到: contribution_summary.csv")
    
    # 可视化平均贡献
    visualize_average_contribution(contrib_summary)

# 可视化平均贡献
def visualize_average_contribution(contrib_summary):
    print("\n生成平均特征贡献可视化...")
    
    # 取前20个平均贡献最大的特征
    top_contrib = contrib_summary.head(20)
    
    # 创建水平条形图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制平均贡献
    bars = ax.barh(range(len(top_contrib)), top_contrib['average_contribution'], 
                   xerr=top_contrib['std_contribution'], color=['#27ae60' if x > 0 else '#e74c3c' for x in top_contrib['average_contribution']])
    
    # 设置y轴标签
    ax.set_yticks(range(len(top_contrib)))
    ax.set_yticklabels(top_contrib['feature'])
    
    # 设置x轴标签
    ax.set_xlabel('平均贡献值')
    
    # 设置标题
    ax.set_title('特征平均贡献值（前20名）')
    
    # 反转y轴，使重要性高的特征在顶部
    ax.invert_yaxis()
    
    # 添加数值标签
    for i, v in enumerate(top_contrib['average_contribution']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(base_path, 'average_contribution_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"平均特征贡献图已保存到: {output_path}")

# 主函数
def main():
    print("=== 潜在高价值客户预测 - 模型解释 ===")
    
    # 加载数据
    df_train, df_test = load_data()
    
    # 准备特征和目标变量
    X_train, X_test, y_train, y_test, feature_cols, df_train_full, df_test_full = prepare_features(df_train, df_test)
    
    # 训练LightGBM模型
    gbm = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # 评估模型
    y_pred, y_pred_binary = evaluate_model(gbm, X_test, y_test)
    
    # 全局解释
    importance_df = global_explanation(gbm, X_train, feature_cols)
    
    # 局部解释
    local_explanation(gbm, X_test, feature_cols, df_test_full, y_pred, y_pred_binary)
    
    print("\n=== 模型解释完成 ===")
    print(f"生成的文件:")
    print(f"- 特征重要性: feature_importance.csv")
    print(f"- 特征重要性可视化: feature_importance_plot.png")
    print(f"- 特征重要性对比: feature_importance_comparison.png")
    print(f"- 特征贡献摘要: contribution_summary.csv")
    print(f"- 平均特征贡献图: average_contribution_plot.png")
    print(f"- 客户特征贡献文件: customer_*_contribution.csv")
    print(f"- 客户贡献可视化: customer_*_contribution_waterfall.png")

if __name__ == "__main__":
    main()
