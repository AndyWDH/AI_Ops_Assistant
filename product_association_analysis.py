import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """加载并预处理数据"""
    # 读取数据
    file_path = 'E:\\AI\\AI大模型应用第17期\\AI大模型应用第17期\\25-项目实战：AI运营助手\\CASE-百万客群经营 20251224\\customer_behavior_assets.csv'
    df = pd.read_csv(file_path)
    
    # 按customer_id聚合，获取每个客户的产品持有情况
    # 对每个产品字段，只要客户有一条记录持有该产品，就认为客户持有该产品
    customer_product_df = df.groupby('customer_id').agg({
        'deposit_flag': 'max',
        'financial_flag': 'max', 
        'fund_flag': 'max',
        'insurance_flag': 'max'
    }).reset_index()
    
    # 重命名字段，使结果更直观
    product_mapping = {
        'deposit_flag': '存款',
        'financial_flag': '理财',
        'fund_flag': '基金',
        'insurance_flag': '保险'
    }
    customer_product_df = customer_product_df.rename(columns=product_mapping)
    
    return customer_product_df

def generate_item_sets(customer_product_df):
    """生成事务数据集"""
    # 获取产品列表
    products = ['存款', '理财', '基金', '保险']
    
    # 生成事务列表，每个事务是客户持有的产品集合
    transactions = []
    for _, row in customer_product_df.iterrows():
        transaction = [product for product in products if row[product] == 1]
        if transaction:  # 只保留至少持有一种产品的客户
            transactions.append(transaction)
    
    return transactions

def create_binary_matrix(customer_product_df):
    """创建用于Apriori算法的二进制矩阵"""
    return customer_product_df.drop('customer_id', axis=1)

def run_apriori_analysis(binary_matrix, min_support=0.1, min_confidence=0.5):
    """运行Apriori算法，生成频繁项集和关联规则"""
    # 生成频繁项集
    frequent_itemsets = apriori(binary_matrix, min_support=min_support, use_colnames=True)
    
    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    return frequent_itemsets, rules

def visualize_frequent_itemsets(frequent_itemsets):
    """可视化频繁项集"""
    plt.figure(figsize=(12, 6))
    
    # 按支持度排序
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
    
    # 提取项集名称和支持度
    itemsets = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    support = frequent_itemsets['support']
    
    # 绘制条形图
    sns.barplot(x=support, y=itemsets, palette='viridis')
    plt.title('频繁产品组合支持度', fontsize=14)
    plt.xlabel('支持度', fontsize=12)
    plt.ylabel('产品组合', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('frequent_itemsets.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_association_rules(rules):
    """可视化关联规则"""
    plt.figure(figsize=(12, 8))
    
    # 提取规则信息
    antecedents = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    consequents = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    confidence = rules['confidence']
    lift = rules['lift']
    
    # 创建散点图，x轴为置信度，y轴为提升度，点大小为支持度
    scatter = plt.scatter(confidence, lift, s=rules['support']*1000, 
                         c=rules['support'], cmap='viridis', alpha=0.7)
    
    # 添加标签
    for i, (antecedent, consequent) in enumerate(zip(antecedents, consequents)):
        plt.annotate(f'{antecedent} → {consequent}', 
                    (confidence[i], lift[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, alpha=0.7)
    
    plt.title('关联规则分析', fontsize=14)
    plt.xlabel('置信度', fontsize=12)
    plt.ylabel('提升度', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(scatter, label='支持度')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('association_rules.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_product_combinations():
    """主函数：分析产品组合"""
    print("开始产品组合关联分析...")
    
    # 1. 加载并预处理数据
    customer_product_df = load_and_preprocess_data()
    print(f"预处理完成，共 {len(customer_product_df)} 个客户")
    
    # 2. 生成事务数据集
    transactions = generate_item_sets(customer_product_df)
    print(f"生成 {len(transactions)} 条事务记录")
    
    # 3. 创建二进制矩阵
    binary_matrix = create_binary_matrix(customer_product_df)
    
    # 4. 运行Apriori分析
    frequent_itemsets, rules = run_apriori_analysis(binary_matrix, min_support=0.1, min_confidence=0.3)
    
    print(f"\n频繁项集结果 ({len(frequent_itemsets)} 个):")
    print(frequent_itemsets.sort_values('support', ascending=False).to_string(index=False))
    
    print(f"\n关联规则结果 ({len(rules)} 个):")
    # 选择关键列并排序
    rules_sorted = rules.sort_values(['lift', 'confidence'], ascending=False)
    print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))
    
    # 5. 可视化结果
    if not frequent_itemsets.empty:
        visualize_frequent_itemsets(frequent_itemsets)
        print("\n频繁项集可视化已保存为 frequent_itemsets.png")
    
    if not rules.empty:
        visualize_association_rules(rules)
        print("关联规则可视化已保存为 association_rules.png")
    
    # 6. 生成详细分析报告
    generate_analysis_report(customer_product_df, frequent_itemsets, rules)
    
    print("\n产品组合关联分析完成！")

def generate_analysis_report(customer_product_df, frequent_itemsets, rules):
    """生成详细分析报告"""
    report = "# 产品组合关联分析报告\n\n"
    
    # 1. 数据概览
    report += "## 1. 数据概览\n"
    report += f"- 分析客户数: {len(customer_product_df)}\n"
    
    # 产品持有情况统计
    products = ['存款', '理财', '基金', '保险']
    product_holdings = customer_product_df[products].sum().sort_values(ascending=False)
    report += "- 产品持有情况:\n"
    for product, count in product_holdings.items():
        report += f"  * {product}: {count} 人 ({count/len(customer_product_df)*100:.1f}%)\n"
    
    # 2. 频繁项集分析
    report += "\n## 2. 频繁产品组合\n"
    report += "频繁产品组合是指同时被较多客户持有的产品集合。\n\n"
    
    # 按项集大小分组
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    for length in sorted(frequent_itemsets['length'].unique()):
        report += f"### {length}项产品组合\n"
        subset = frequent_itemsets[frequent_itemsets['length'] == length].sort_values('support', ascending=False)
        if not subset.empty:
            for _, row in subset.iterrows():
                items = ', '.join(list(row['itemsets']))
                report += f"- {items}: 支持度 {row['support']:.3f}\n"
        else:
            report += "- 无\n"
    
    # 3. 关联规则分析
    report += "\n## 3. 关联规则\n"
    report += "关联规则表示产品之间的关联关系，格式为 ' antecedents → consequents '。\n\n"
    
    if not rules.empty:
        rules_sorted = rules.sort_values(['lift', 'confidence'], ascending=False)
        report += "| 规则 | 支持度 | 置信度 | 提升度 |\n"
        report += "|------|--------|--------|--------|\n"
        for _, row in rules_sorted.iterrows():
            antecedent = ', '.join(list(row['antecedents']))
            consequent = ', '.join(list(row['consequents']))
            rule = f"{antecedent} → {consequent}"
            report += f"| {rule} | {row['support']:.3f} | {row['confidence']:.3f} | {row['lift']:.3f} |\n"
    else:
        report += "- 无关联规则\n"
    
    # 4. 业务建议
    report += "\n## 4. 业务建议\n"
    report += "根据关联分析结果，提出以下业务建议：\n\n"
    
    # 基于关联规则生成建议
    if not rules.empty:
        rules_sorted = rules.sort_values('lift', ascending=False)
        top_rules = rules_sorted.head(5)
        
        for _, row in top_rules.iterrows():
            antecedent = ', '.join(list(row['antecedents']))
            consequent = ', '.join(list(row['consequents']))
            report += f"- **交叉销售建议**：对于持有 {antecedent} 的客户，可以推荐 {consequent}，\n"
            report += f"  该组合的置信度为 {row['confidence']:.1%}，提升度为 {row['lift']:.2f}，\n"
            report += f"  意味着推荐成功的概率比随机推荐高 {row['lift']:.2f} 倍。\n\n"
    
    # 基于频繁项集生成建议
    frequent_itemsets_sorted = frequent_itemsets.sort_values('support', ascending=False)
    top_combined = frequent_itemsets_sorted[frequent_itemsets_sorted['length'] >= 2].head(3)
    
    if not top_combined.empty:
        report += "- **组合产品设计**：考虑设计包含以下产品的组合套餐：\n"
        for _, row in top_combined.iterrows():
            items = ', '.join(list(row['itemsets']))
            report += f"  * {items}（支持度 {row['support']:.1%}）\n"
    
    # 保存报告
    with open('产品组合关联分析报告.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析报告已保存为 产品组合关联分析报告.md")

if __name__ == "__main__":
    analyze_product_combinations()