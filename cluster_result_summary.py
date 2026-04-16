import pandas as pd
import os

# 获取当前目录
base_path = os.path.dirname(__file__)

# 读取聚类分析结果
def read_cluster_analysis():
    cluster_df = pd.read_csv(os.path.join(base_path, 'advanced_cluster_analysis.csv'))
    return cluster_df

# 为聚类群组分配更合适的名称
def assign_better_names(cluster_df):
    # 根据特征为每个群组分配更准确的名称
    better_names = {
        '客户群2': '普通客户群1',
        '客户群3': '中产稳健客户',
        '客户群4': '普通客户群2',
        '客户群5': '普通客户群3',
        '客户群6': '中产高资产客户',
        '客户群8': '高复购客户',
        '年轻高消费客户': '年轻高消费客户',
        '高资产客户': '高资产客户'
    }
    
    # 应用新名称
    cluster_df['better_cluster_name'] = cluster_df['cluster_name'].map(better_names)
    
    return cluster_df

# 分析每个群组的特征
def analyze_cluster_features(cluster_df):
    print("=== 客户聚类分析结果 ===")
    
    for idx, row in cluster_df.iterrows():
        cluster_name = row['better_cluster_name']
        size = int(row['cluster_size'])
        percentage = row['cluster_percentage']
        
        print(f"\n{cluster_name}:")
        print(f"  数量: {size}人 ({percentage:.2f}%)")
        print(f"  平均年龄: {row['age']:.1f}岁")
        print(f"  平均月收入: {row['monthly_income']:.0f}元")
        print(f"  平均总资产: {row['total_assets']:.0f}元")
        print(f"  平均信用卡月支出: {row['credit_card_monthly_expense']:.0f}元")
        print(f"  平均金融复购次数: {row['financial_repurchase_count']:.2f}次")
        print(f"  平均产品持有数: {row['product_count']:.2f}个")
        print(f"  平均APP登录次数: {row['app_login_count']:.1f}次")
        print(f"  平均APP金融浏览时长: {row['app_financial_view_time']:.0f}秒")
    
    # 保存带新名称的聚类分析结果
    cluster_df.to_csv(os.path.join(base_path, 'final_cluster_analysis.csv'), index=False, encoding='utf-8')
    print(f"\n带新名称的聚类分析结果已保存到: final_cluster_analysis.csv")

# 主函数
def main():
    # 读取聚类分析结果
    cluster_df = read_cluster_analysis()
    
    # 为聚类群组分配更合适的名称
    cluster_df = assign_better_names(cluster_df)
    
    # 分析每个群组的特征
    analyze_cluster_features(cluster_df)

if __name__ == "__main__":
    main()