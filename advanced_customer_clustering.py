import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取当前目录
base_path = os.path.dirname(__file__)

# 读取数据
def load_data():
    print("加载数据...")
    
    # 读取客户基础信息表
    base_df = pd.read_csv(os.path.join(base_path, 'customer_base.csv'))
    print(f"客户基础信息表 shape: {base_df.shape}")
    
    # 读取客户行为资产表
    behavior_df = pd.read_csv(os.path.join(base_path, 'customer_behavior_assets.csv'))
    print(f"客户行为资产表 shape: {behavior_df.shape}")
    
    return base_df, behavior_df

# 数据预处理和合并
def preprocess_data(base_df, behavior_df):
    print("\n数据预处理和合并...")
    
    # 选择最新月份的行为资产数据（2025-04）
    latest_behavior = behavior_df[behavior_df['stat_month'] == '2025-04']
    print(f"最新月份（2025-04）行为资产数据 shape: {latest_behavior.shape}")
    
    # 合并两个表
    merged_df = pd.merge(base_df, latest_behavior, on='customer_id', how='inner')
    print(f"合并后数据 shape: {merged_df.shape}")
    
    # 选择用于聚类的关键特征，针对用户需求优化
    selected_features = [
        # 人口统计学特征
        'age', 'gender', 'occupation_type', 'marriage_status', 'monthly_income', 'city_level',
        # 资产特征
        'total_assets', 'deposit_balance', 'financial_balance', 'fund_balance', 'insurance_balance',
        # 行为特征
        'product_count', 'financial_repurchase_count', 'credit_card_monthly_expense',
        'investment_monthly_count', 'app_login_count', 'app_financial_view_time'
    ]
    
    # 提取选定特征
    df = merged_df[selected_features].copy()
    
    # 编码类别特征
    label_encoders = {}
    categorical_cols = ['gender', 'occupation_type', 'marriage_status', 'city_level']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # 选择最终用于聚类的数值特征
    numeric_features = [
        'age', 'monthly_income', 'total_assets', 'deposit_balance', 'financial_balance',
        'fund_balance', 'insurance_balance', 'product_count', 'financial_repurchase_count',
        'credit_card_monthly_expense', 'investment_monthly_count', 'app_login_count',
        'app_financial_view_time'
    ] + [col + '_encoded' for col in categorical_cols]
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_features])
    
    print(f"\n用于聚类的特征: {numeric_features}")
    print(f"特征数量: {len(numeric_features)}")
    print(f"标准化后数据 shape: {scaled_data.shape}")
    
    return merged_df, df, scaled_data, numeric_features, scaler

# 执行聚类分析（使用指定的K值）
def perform_clustering(scaled_data, merged_df, k_value, numeric_features):
    print(f"\n使用 K-means 算法执行聚类分析（K={k_value}）...")
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # 将聚类结果添加到原始数据
    merged_df['cluster_label'] = cluster_labels
    
    # 计算轮廓系数
    sil_score = silhouette_score(scaled_data, cluster_labels)
    print(f"聚类结果轮廓系数: {sil_score:.4f}")
    
    return merged_df, kmeans

# 为聚类分配符合用户需求的名称
def assign_cluster_names(merged_df):
    print("\n为聚类分配名称...")
    
    # 计算每个聚类的关键特征均值，确保包含所有用于命名的特征
    cluster_stats = merged_df.groupby('cluster_label').agg({
        'age': 'mean',
        'monthly_income': 'mean',
        'total_assets': 'mean',
        'financial_repurchase_count': 'mean',
        'credit_card_monthly_expense': 'mean',
        'marriage_status': lambda x: x.value_counts().idxmax(),
        'product_count': 'mean',
        'investment_monthly_count': 'mean',
        'financial_balance': 'mean',
        'app_login_count': 'mean',
        'app_financial_view_time': 'mean'
    }).reset_index()
    
    # 基于特征均值分配群组名称
    cluster_names = {}
    
    for idx, row in cluster_stats.iterrows():
        label = row['cluster_label']
        
        # 年轻高消费客户（年龄<35，信用卡月支出>10000）
        if row['age'] < 35 and row['credit_card_monthly_expense'] > 10000:
            cluster_names[label] = '年轻高消费客户'
        # 高复购客户（复购次数>5，产品持有数>3）
        elif row['financial_repurchase_count'] > 5 and row['product_count'] > 3:
            cluster_names[label] = '高复购客户'
        # 中产家庭（已婚，月收入15000-30000，总资产50-150万）
        elif row['marriage_status'] == '已婚' and 15000 <= row['monthly_income'] <= 30000 and 500000 <= row['total_assets'] <= 1500000:
            cluster_names[label] = '中产家庭客户'
        # 高资产客户（总资产>200万）
        elif row['total_assets'] > 2000000:
            cluster_names[label] = '高资产客户'
        # 投资活跃客户（投资月均次数>3，金融产品余额>50万）
        elif row['investment_monthly_count'] > 3 and row['financial_balance'] > 500000:
            cluster_names[label] = '投资活跃客户'
        # 稳健储蓄客户（存款余额>总资产的60%）
        elif merged_df[merged_df['cluster_label'] == label]['deposit_balance'].mean() > merged_df[merged_df['cluster_label'] == label]['total_assets'].mean() * 0.6:
            cluster_names[label] = '稳健储蓄客户'
        # 活跃数字化客户（APP登录次数>20，浏览时长>1000秒）
        elif row['app_login_count'] > 20 and row['app_financial_view_time'] > 1000:
            cluster_names[label] = '活跃数字化客户'
        # 基础客户（资产较少，活跃度低）
        elif row['total_assets'] < 100000 and row['app_login_count'] < 5:
            cluster_names[label] = '基础客户'
        # 其他客户
        else:
            cluster_names[label] = f'客户群{label+1}'
    
    # 将聚类名称添加到数据中
    merged_df['cluster_name'] = merged_df['cluster_label'].map(cluster_names)
    
    # 显示聚类名称统计
    print("\n聚类名称分配结果:")
    cluster_counts = merged_df['cluster_name'].value_counts().sort_values(ascending=False)
    for name, count in cluster_counts.items():
        print(f"{name}: {count}人 ({count/len(merged_df)*100:.2f}%)")
    
    return merged_df, cluster_names

# 分析聚类结果
def analyze_clusters(merged_df):
    print("\n分析聚类结果...")
    
    # 只使用merged_df中存在的关键特征进行分析
    key_features = ['age', 'monthly_income', 'total_assets', 'financial_repurchase_count', 
                   'credit_card_monthly_expense', 'product_count', 'investment_monthly_count',
                   'financial_balance', 'app_login_count', 'app_financial_view_time']
    
    # 计算每个聚类的特征均值
    cluster_analysis = merged_df.groupby('cluster_name')[key_features].mean()
    
    # 添加聚类大小
    cluster_sizes = merged_df['cluster_name'].value_counts()
    cluster_analysis['cluster_size'] = cluster_sizes
    cluster_analysis['cluster_percentage'] = (cluster_sizes / len(merged_df)) * 100
    
    # 保存聚类分析结果
    cluster_analysis.to_csv(os.path.join(base_path, 'advanced_cluster_analysis.csv'), encoding='utf-8')
    print(f"聚类分析结果已保存到: advanced_cluster_analysis.csv")
    
    # 显示聚类分析结果（关键特征）
    print("\n聚类关键特征分析:")
    display_features = ['age', 'monthly_income', 'total_assets', 'financial_repurchase_count', 'credit_card_monthly_expense']
    print(cluster_analysis[display_features + ['cluster_size', 'cluster_percentage']])
    
    return cluster_analysis

# 可视化聚类结果
def visualize_clusters(merged_df, scaled_data):
    print("\n可视化聚类结果...")
    
    # 使用PCA降维到2维进行可视化
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled_data)
    
    # 添加PCA结果到数据
    merged_df['pca1'] = pca_result[:, 0]
    merged_df['pca2'] = pca_result[:, 1]
    
    # 绘制PCA聚类图
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x='pca1', y='pca2',
        hue='cluster_name',
        data=merged_df,
        palette='viridis',
        s=100,
        alpha=0.7,
        legend='full'
    )
    plt.title('客户聚类PCA可视化', fontsize=16)
    plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} 方差解释)', fontsize=12)
    plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} 方差解释)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'advanced_cluster_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA聚类可视化图已保存到: advanced_cluster_pca.png")
    
    # 绘制聚类分布饼图
    plt.figure(figsize=(12, 8))
    cluster_counts = merged_df['cluster_name'].value_counts()
    plt.pie(
        cluster_counts.values,
        labels=cluster_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('viridis', len(cluster_counts))
    )
    plt.title('客户聚类分布', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'advanced_cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"聚类分布饼图已保存到: advanced_cluster_distribution.png")
    
    # 绘制关键特征箱线图
    key_features = ['age', 'monthly_income', 'total_assets', 'financial_repurchase_count', 'credit_card_monthly_expense']
    
    for feature in key_features:
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            x='cluster_name',
            y=feature,
            data=merged_df,
            palette='viridis'
        )
        plt.title(f'不同客户群的{feature}分布', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylabel(feature, fontsize=12)
        plt.xlabel('客户群', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'advanced_cluster_{feature}_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{feature}箱线图已保存到: advanced_cluster_{feature}_boxplot.png")

# 主函数
def main():
    print("=== 高级客户聚类分析 ===")
    
    # 加载数据
    base_df, behavior_df = load_data()
    
    # 数据预处理和合并
    merged_df, df, scaled_data, numeric_features, scaler = preprocess_data(base_df, behavior_df)
    
    # 使用K=8进行聚类，确保能够生成多种客户群
    k_value = 8
    merged_df, kmeans = perform_clustering(scaled_data, merged_df, k_value, numeric_features)
    
    # 为聚类分配符合用户需求的名称
    merged_df, cluster_names = assign_cluster_names(merged_df)
    
    # 分析聚类结果
    cluster_analysis = analyze_clusters(merged_df)
    
    # 可视化聚类结果
    visualize_clusters(merged_df, scaled_data)
    
    # 保存带聚类结果的完整数据
    merged_df.to_csv(os.path.join(base_path, 'advanced_customer_clusters.csv'), index=False, encoding='utf-8')
    print(f"\n带聚类结果的客户数据已保存到: advanced_customer_clusters.csv")
    
    print("\n=== 高级客户聚类分析完成 ===")
    print(f"\n最终聚类结果:")
    for name, count in merged_df['cluster_name'].value_counts().items():
        percentage = count / len(merged_df) * 100
        print(f"{name}: {count}人 ({percentage:.2f}%)")

if __name__ == "__main__":
    main()