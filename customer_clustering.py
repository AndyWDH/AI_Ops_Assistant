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
    
    # 选择用于聚类的特征
    # 数值特征
    numeric_features = ['age', 'monthly_income', 'total_assets', 'deposit_balance', 
                       'financial_balance', 'fund_balance', 'insurance_balance',
                       'product_count', 'financial_repurchase_count', 'credit_card_monthly_expense',
                       'investment_monthly_count', 'app_login_count', 'app_financial_view_time']
    
    # 类别特征（需要编码）
    categorical_features = ['gender', 'occupation_type', 'marriage_status', 'city_level']
    
    # 检查数值特征缺失值
    print(f"\n数值特征缺失值情况:")
    print(merged_df[numeric_features].isnull().sum())
    
    # 检查类别特征缺失值
    print(f"\n类别特征缺失值情况:")
    print(merged_df[categorical_features].isnull().sum())
    
    # 处理缺失值（此处数据质量较好，无缺失值）
    
    # 编码类别特征
    le = LabelEncoder()
    for col in categorical_features:
        merged_df[col + '_encoded'] = le.fit_transform(merged_df[col])
    
    # 选择最终用于聚类的特征
    cluster_features = numeric_features + [col + '_encoded' for col in categorical_features]
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_df[cluster_features])
    
    print(f"\n用于聚类的特征数: {len(cluster_features)}")
    print(f"标准化后数据 shape: {scaled_data.shape}")
    
    return merged_df, scaled_data, cluster_features, scaler

# 确定最佳聚类数量
def determine_optimal_clusters(scaled_data, max_clusters=10):
    print(f"\n确定最佳聚类数量（1-{max_clusters}）...")
    
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
        print(f"K={k} - 轮廓系数: {silhouette_scores[-1]:.4f}, 畸变值: {inertias[-1]:.2f}")
    
    # 绘制肘部图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters+1), inertias, marker='o')
    plt.xlabel('聚类数量 K')
    plt.ylabel('畸变值 (Inertia)')
    plt.title('肘部法确定最佳 K 值')
    
    # 绘制轮廓系数图
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
    plt.xlabel('聚类数量 K')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数法确定最佳 K 值')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'optimal_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n最佳聚类数量分析图已保存到: optimal_clusters.png")
    
    # 选择轮廓系数最大的 K 值
    optimal_k = np.argmax(silhouette_scores) + 2
    print(f"\n最佳聚类数量: {optimal_k}")
    
    return optimal_k

# 执行聚类分析
def perform_clustering(scaled_data, merged_df, optimal_k, cluster_features):
    print(f"\n使用 K-means 算法执行聚类分析（K={optimal_k}）...")
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # 将聚类结果添加到原始数据
    merged_df['cluster_label'] = cluster_labels
    
    # 计算轮廓系数
    sil_score = silhouette_score(scaled_data, cluster_labels)
    print(f"聚类结果轮廓系数: {sil_score:.4f}")
    
    return merged_df, kmeans

# 分析聚类结果
def analyze_clusters(merged_df, cluster_features):
    print("\n分析聚类结果...")
    
    # 计算每个聚类的特征均值
    cluster_analysis = merged_df.groupby('cluster_label')[cluster_features].mean()
    
    # 添加聚类大小
    cluster_sizes = merged_df['cluster_label'].value_counts().sort_index()
    cluster_analysis['cluster_size'] = cluster_sizes
    
    # 保存聚类分析结果
    cluster_analysis.to_csv(os.path.join(base_path, 'cluster_analysis.csv'), encoding='utf-8')
    print(f"聚类分析结果已保存到: cluster_analysis.csv")
    
    # 显示聚类分析结果
    print("\n聚类特征均值分析:")
    print(cluster_analysis)
    
    # 生成聚类名称
    cluster_names = assign_cluster_names(cluster_analysis)
    merged_df['cluster_name'] = merged_df['cluster_label'].map(cluster_names)
    
    # 保存带聚类结果的数据
    merged_df.to_csv(os.path.join(base_path, 'customer_clusters.csv'), index=False, encoding='utf-8')
    print(f"带聚类结果的客户数据已保存到: customer_clusters.csv")
    
    return merged_df, cluster_names

# 为聚类分配名称
def assign_cluster_names(cluster_analysis):
    # 根据聚类特征均值分配名称
    cluster_names = {}
    
    for i in range(len(cluster_analysis)):
        cluster = cluster_analysis.iloc[i]
        
        # 基于关键特征定义聚类名称
        if cluster['total_assets'] > 1000000 and cluster['financial_repurchase_count'] > 10:
            cluster_names[i] = '高复购高资产客户'
        elif cluster['total_assets'] > 500000 and cluster['age'] > 30 and cluster['marriage_status_encoded'] == 0:
            cluster_names[i] = '中产家庭客户'
        elif cluster['age'] < 30 and cluster['credit_card_monthly_expense'] > 5000:
            cluster_names[i] = '年轻高消费客户'
        elif cluster['financial_balance'] > 300000 and cluster['fund_balance'] > 200000:
            cluster_names[i] = '投资偏好客户'
        elif cluster['deposit_balance'] > 500000:
            cluster_names[i] = '稳健储蓄客户'
        elif cluster['app_login_count'] > 30 and cluster['app_financial_view_time'] > 1000:
            cluster_names[i] = '活跃数字化客户'
        elif cluster['monthly_income'] < 5000 and cluster['total_assets'] < 100000:
            cluster_names[i] = '基础客户'
        else:
            cluster_names[i] = f'客户群{i+1}'
    
    return cluster_names

# 可视化聚类结果
def visualize_clusters(merged_df, scaled_data, kmeans, cluster_features):
    print("\n可视化聚类结果...")
    
    # 使用PCA降维到2维进行可视化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # 添加PCA结果到数据
    merged_df['pca1'] = pca_result[:, 0]
    merged_df['pca2'] = pca_result[:, 1]
    
    # 绘制PCA聚类图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(merged_df['pca1'], merged_df['pca2'], 
                         c=merged_df['cluster_label'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='聚类标签')
    plt.title('客户聚类PCA可视化')
    plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} 方差解释)')
    plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} 方差解释)')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'cluster_pca_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA聚类可视化图已保存到: cluster_pca_visualization.png")
    
    # 绘制聚类分布饼图
    plt.figure(figsize=(10, 8))
    cluster_counts = merged_df['cluster_name'].value_counts()
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('客户聚类分布')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"聚类分布饼图已保存到: cluster_distribution.png")
    
    # 绘制关键特征箱线图
    key_features = ['age', 'monthly_income', 'total_assets', 'financial_repurchase_count', 'credit_card_monthly_expense']
    
    for feature in key_features:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='cluster_name', y=feature, data=merged_df)
        plt.title(f'不同聚类的{feature}分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'cluster_{feature}_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{feature}箱线图已保存到: cluster_{feature}_boxplot.png")

# 主函数
def main():
    print("=== 客户聚类分析 ===")
    
    # 加载数据
    base_df, behavior_df = load_data()
    
    # 数据预处理和合并
    merged_df, scaled_data, cluster_features, scaler = preprocess_data(base_df, behavior_df)
    
    # 确定最佳聚类数量
    optimal_k = determine_optimal_clusters(scaled_data, max_clusters=8)
    
    # 执行聚类分析
    merged_df, kmeans = perform_clustering(scaled_data, merged_df, optimal_k, cluster_features)
    
    # 分析聚类结果
    merged_df, cluster_names = analyze_clusters(merged_df, cluster_features)
    
    # 可视化聚类结果
    visualize_clusters(merged_df, scaled_data, kmeans, cluster_features)
    
    print("\n=== 客户聚类分析完成 ===")
    print(f"\n聚类结果:")
    for label, name in cluster_names.items():
        size = len(merged_df[merged_df['cluster_label'] == label])
        print(f"聚类 {label}: {name} (数量: {size})")

if __name__ == "__main__":
    main()