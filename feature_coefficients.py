import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

# 设置文件路径
base_path = os.path.dirname(__file__)
train_path = os.path.join(base_path, 'train_features.csv')
test_path = os.path.join(base_path, 'test_features.csv')
model_path = os.path.join(base_path, 'time_based_model.json')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
def load_data():
    print("开始加载数据...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 分离特征和目标变量
    X_train = df_train.drop(['target_high_value', 'customer_id', 'stat_month'], axis=1)
    y_train = df_train['target_high_value']
    
    X_test = df_test.drop(['target_high_value', 'customer_id', 'stat_month'], axis=1)
    y_test = df_test['target_high_value']
    
    print(f"训练集形状：{X_train.shape}")
    print(f"测试集形状：{X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# 加载模型
def load_model():
    print("\n加载XGBoost模型...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("模型加载完成")
    return model

# 特征重要性分析（使用XGBoost内置功能）
def analyze_feature_importance(model, X_train, X_test):
    print("\n特征重要性分析...")
    
    # 1. 使用feature_importance_属性获取特征重要性
    feature_importance = model.feature_importances_
    
    # 2. 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 3. 计算特征与目标变量的相关性，用于展示正负影响
    print("\n计算特征与目标变量的相关性...")
    # 获取训练数据的特征和目标变量
    df_train_full = pd.read_csv(train_path)
    
    # 计算皮尔逊相关系数
    correlations = df_train_full.corr(numeric_only=True)['target_high_value'].reset_index()
    correlations.columns = ['feature', 'correlation']
    
    # 只保留与特征重要性数据匹配的特征
    correlations = correlations[correlations['feature'].isin(importance_df['feature'])].copy()
    
    # 4. 合并特征重要性和相关性
    importance_corr = pd.merge(importance_df, correlations, on='feature', how='left')
    
    # 计算影响方向（基于相关性）
    importance_corr['impact'] = importance_corr['correlation'].apply(lambda x: "正影响" if x > 0 else "负影响" if x < 0 else "无影响")
    
    # 显示前20个重要特征及其正负影响
    print("\n前20个重要特征的影响分析（基于相关性）：")
    print("特征名称                              | 重要性分数    | 相关系数    | 影响方向")
    print("-" * 80)
    
    for _, row in importance_corr.head(20).iterrows():
        print(f"{row['feature']:40} | {row['importance']:10.6f} | {row['correlation']:10.6f} | {row['impact']}")
    
    # 5. 可视化特征重要性和正负影响
    visualize_feature_importance(importance_corr.head(20))
    
    return importance_corr

# 可视化特征重要性和正负影响
def visualize_feature_importance(importance_corr):
    print("\n生成特征重要性可视化...")
    
    # 创建水平条形图
    plt.figure(figsize=(12, 8))
    
    # 根据相关性设置颜色（正影响：蓝色，负影响：红色）
    colors = ['skyblue' if corr > 0 else 'salmon' for corr in importance_corr['correlation']]
    
    # 绘制条形图
    bars = plt.barh(range(len(importance_corr)), importance_corr['importance'], color=colors)
    
    # 设置y轴标签
    plt.yticks(range(len(importance_corr)), importance_corr['feature'])
    
    # 设置x轴标签
    plt.xlabel('特征重要性分数')
    
    # 设置标题
    plt.title('XGBoost模型特征重要性及影响方向（前20名）')
    
    # 反转y轴，使重要性高的特征在顶部
    plt.gca().invert_yaxis()
    
    # 添加数值标签
    for i, v in enumerate(importance_corr['importance']):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    # 添加图例
    handles = [
        plt.Rectangle((0,0),1,1, color='skyblue', label='正影响'),
        plt.Rectangle((0,0),1,1, color='salmon', label='负影响')
    ]
    plt.legend(handles=handles, loc='lower right')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(base_path, 'feature_importance_with_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存到：{output_path}")

# 主函数
def main():
    print("=== XGBoost模型特征重要性及影响分析 ===")
    
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    
    # 加载模型
    model = load_model()
    
    # 特征重要性分析
    importance_corr = analyze_feature_importance(model, X_train, X_test)
    
    print("\n=== 分析完成 ===")
    print("已生成的文件：")
    print(f"- 特征重要性及影响方向图：{os.path.join(base_path, 'feature_importance_with_impact.png')}")
    print(f"- 模型文件：{model_path}")

if __name__ == "__main__":
    main()