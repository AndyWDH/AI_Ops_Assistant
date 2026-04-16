import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def generate_transactions(customer_product_df):
    """生成事务数据集"""
    products = ['存款', '理财', '基金', '保险']
    transactions = []
    
    for _, row in customer_product_df.iterrows():
        transaction = [product for product in products if row[product] == 1]
        if transaction:  # 只保留至少持有一种产品的客户
            transactions.append(transaction)
    
    return transactions

def calculate_support(itemset, transactions):
    """计算项集的支持度"""
    count = 0
    for transaction in transactions:
        if set(itemset).issubset(set(transaction)):
            count += 1
    return count / len(transactions)

def generate_candidates(prev_itemsets, k):
    """生成候选k项集"""
    candidates = set()
    n = len(prev_itemsets)
    
    # 两两合并前一轮的频繁项集
    for i in range(n):
        for j in range(i+1, n):
            # 获取两个项集
            itemset1 = sorted(list(prev_itemsets[i]))
            itemset2 = sorted(list(prev_itemsets[j]))
            
            # 检查前k-2个元素是否相同
            if itemset1[:k-2] == itemset2[:k-2]:
                # 合并生成候选k项集
                candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                candidates.add(candidate)
    
    return candidates

def apriori(transactions, min_support):
    """手动实现Apriori算法"""
    # 生成1项候选集
    all_products = set()
    for transaction in transactions:
        all_products.update(transaction)
    
    # 初始候选集为所有单个产品
    candidates = [(product,) for product in all_products]
    frequent_itemsets = {}
    k = 1
    
    while candidates:
        # 计算每个候选集的支持度
        itemset_support = {}
        for candidate in candidates:
            support = calculate_support(candidate, transactions)
            if support >= min_support:
                itemset_support[candidate] = support
        
        # 如果没有频繁项集，退出循环
        if not itemset_support:
            break
        
        # 保存当前频繁项集
        frequent_itemsets[k] = itemset_support
        
        # 生成下一轮候选集
        k += 1
        candidates = list(generate_candidates(list(itemset_support.keys()), k))
    
    return frequent_itemsets

def generate_rules(frequent_itemsets, transactions, min_confidence):
    """生成关联规则"""
    rules = []
    
    # 遍历所有频繁项集（从2项集开始）
    for k in frequent_itemsets:
        if k < 2:
            continue
        
        for itemset in frequent_itemsets[k]:
            itemset_support = frequent_itemsets[k][itemset]
            
            # 生成所有可能的规则前提（非空真子集）
            for i in range(1, k):
                # 生成所有i项子集作为前提
                for antecedent in combinations(itemset, i):
                    antecedent = tuple(sorted(antecedent))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    
                    # 计算置信度
                    antecedent_support = frequent_itemsets[len(antecedent)][antecedent]
                    confidence = itemset_support / antecedent_support
                    
                    if confidence >= min_confidence:
                        # 计算提升度
                        consequent_support = frequent_itemsets[len(consequent)][consequent]
                        lift = confidence / consequent_support
                        
                        rules.append({
                            'antecedents': antecedent,
                            'consequents': consequent,
                            'support': itemset_support,
                            'confidence': confidence,
                            'lift': lift
                        })
    
    return rules

def visualize_frequent_itemsets(frequent_itemsets):
    """可视化频繁项集"""
    plt.figure(figsize=(12, 8))
    
    # 收集所有频繁项集和支持度
    itemsets = []
    support_values = []
    
    for k in frequent_itemsets:
        for itemset, support in frequent_itemsets[k].items():
            itemsets.append(', '.join(itemset))
            support_values.append(support)
    
    # 按支持度排序
    sorted_indices = np.argsort(support_values)[::-1]
    sorted_itemsets = [itemsets[i] for i in sorted_indices]
    sorted_support = [support_values[i] for i in sorted_indices]
    
    # 绘制条形图
    plt.barh(range(len(sorted_itemsets)), sorted_support, color='skyblue')
    plt.yticks(range(len(sorted_itemsets)), sorted_itemsets, fontsize=10)
    plt.xlabel('支持度', fontsize=12)
    plt.ylabel('产品组合', fontsize=12)
    plt.title('频繁产品组合支持度', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('simple_frequent_itemsets.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_rules(rules):
    """可视化关联规则"""
    if not rules:
        print("没有符合条件的关联规则，无法可视化")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 提取规则信息
    antecedents = [', '.join(rule['antecedents']) for rule in rules]
    consequents = [', '.join(rule['consequents']) for rule in rules]
    confidence = [rule['confidence'] for rule in rules]
    lift = [rule['lift'] for rule in rules]
    support = [rule['support'] for rule in rules]
    
    # 创建散点图
    scatter = plt.scatter(confidence, lift, s=[s*1000 for s in support], 
                         c=support, cmap='viridis', alpha=0.7)
    
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
    plt.savefig('simple_association_rules.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_product_combinations():
    """主函数：分析产品组合"""
    print("开始产品组合关联分析...")
    
    # 1. 加载并预处理数据
    customer_product_df = load_and_preprocess_data()
    print(f"预处理完成，共 {len(customer_product_df)} 个客户")
    
    # 2. 生成事务数据集
    transactions = generate_transactions(customer_product_df)
    print(f"生成 {len(transactions)} 条事务记录")
    
    # 3. 运行Apriori算法
    min_support = 0.1
    min_confidence = 0.3
    print(f"\n使用参数：最小支持度={min_support}，最小置信度={min_confidence}")
    
    frequent_itemsets = apriori(transactions, min_support)
    
    # 4. 生成关联规则
    rules = generate_rules(frequent_itemsets, transactions, min_confidence)
    
    # 5. 输出结果
    print(f"\n频繁项集结果：")
    for k in sorted(frequent_itemsets.keys()):
        print(f"\n{k}项频繁产品组合：")
        itemsets = frequent_itemsets[k]
        # 按支持度排序
        sorted_itemsets = sorted(itemsets.items(), key=lambda x: x[1], reverse=True)
        for itemset, support in sorted_itemsets:
            print(f"- {', '.join(itemset)}: 支持度 {support:.3f}")
    
    print(f"\n关联规则结果 ({len(rules)} 条):")
    if rules:
        # 按提升度和置信度排序
        rules_sorted = sorted(rules, key=lambda x: (x['lift'], x['confidence']), reverse=True)
        for rule in rules_sorted:
            antecedent = ', '.join(rule['antecedents'])
            consequent = ', '.join(rule['consequents'])
            print(f"- {antecedent} → {consequent}: 支持度 {rule['support']:.3f}, 置信度 {rule['confidence']:.3f}, 提升度 {rule['lift']:.3f}")
    else:
        print("- 无符合条件的关联规则")
    
    # 6. 可视化结果
    visualize_frequent_itemsets(frequent_itemsets)
    print("\n频繁项集可视化已保存为 simple_frequent_itemsets.png")
    
    visualize_rules(rules)
    print("关联规则可视化已保存为 simple_association_rules.png")
    
    # 7. 生成分析报告
    generate_analysis_report(customer_product_df, frequent_itemsets, rules, transactions)
    
    print("\n产品组合关联分析完成！")

def generate_analysis_report(customer_product_df, frequent_itemsets, rules, transactions):
    """生成详细分析报告"""
    report = "# 产品组合关联分析报告\n\n"
    
    # 1. 数据概览
    report += "## 1. 数据概览\n"
    report += f"- 分析客户数: {len(customer_product_df)}\n"
    report += f"- 有效事务数: {len(transactions)}\n\n"
    
    # 产品持有情况统计
    products = ['存款', '理财', '基金', '保险']
    product_holdings = customer_product_df[products].sum().sort_values(ascending=False)
    report += "- 产品持有情况:\n"
    for product, count in product_holdings.items():
        percentage = count / len(customer_product_df) * 100
        report += f"  * {product}: {count} 人 ({percentage:.1f}%)\n"
    
    # 2. 频繁项集分析
    report += "\n## 2. 频繁产品组合\n"
    report += "频繁产品组合是指同时被较多客户持有的产品集合。\n\n"
    
    for k in sorted(frequent_itemsets.keys()):
        report += f"### {k}项产品组合\n"
        itemsets = frequent_itemsets[k]
        sorted_itemsets = sorted(itemsets.items(), key=lambda x: x[1], reverse=True)
        for itemset, support in sorted_itemsets:
            items = ', '.join(itemset)
            report += f"- {items}: 支持度 {support:.3f}\n"
    
    # 3. 关联规则分析
    report += "\n## 3. 关联规则\n"
    report += "关联规则表示产品之间的关联关系，格式为 ' antecedents → consequents '。\n\n"
    
    if rules:
        rules_sorted = sorted(rules, key=lambda x: (x['lift'], x['confidence']), reverse=True)
        report += "| 规则 | 支持度 | 置信度 | 提升度 |\n"
        report += "|------|--------|--------|--------|\n"
        for rule in rules_sorted:
            antecedent = ', '.join(rule['antecedents'])
            consequent = ', '.join(rule['consequents'])
            rule_str = f"{antecedent} → {consequent}"
            report += f"| {rule_str} | {rule['support']:.3f} | {rule['confidence']:.3f} | {rule['lift']:.3f} |\n"
    else:
        report += "- 无符合条件的关联规则\n"
    
    # 4. 业务建议
    report += "\n## 4. 业务建议\n"
    report += "根据关联分析结果，提出以下业务建议：\n\n"
    
    # 基于关联规则生成建议
    if rules:
        rules_sorted = sorted(rules, key=lambda x: x['lift'], reverse=True)
        top_rules = rules_sorted[:5]  # 取前5条规则
        
        for rule in top_rules:
            antecedent = ', '.join(rule['antecedents'])
            consequent = ', '.join(rule['consequents'])
            report += f"- **交叉销售建议**：对于持有 {antecedent} 的客户，可以推荐 {consequent}，\n"
            report += f"  该组合的置信度为 {rule['confidence']:.1%}，提升度为 {rule['lift']:.2f}，\n"
            report += f"  意味着推荐成功的概率比随机推荐高 {rule['lift']:.2f} 倍。\n\n"
    
    # 基于频繁项集生成建议
    # 提取所有2项及以上的频繁项集
    combined_itemsets = []
    for k in frequent_itemsets:
        if k >= 2:
            for itemset, support in frequent_itemsets[k].items():
                combined_itemsets.append((itemset, support))
    
    # 按支持度排序
    combined_itemsets = sorted(combined_itemsets, key=lambda x: x[1], reverse=True)[:3]  # 取前3个
    
    if combined_itemsets:
        report += "- **组合产品设计**：考虑设计包含以下产品的组合套餐：\n"
        for itemset, support in combined_itemsets:
            items = ', '.join(itemset)
            report += f"  * {items}（支持度 {support:.1%}）\n"
    
    # 保存报告
    with open('产品组合关联分析报告_简易版.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析报告已保存为 产品组合关联分析报告_简易版.md")

if __name__ == "__main__":
    analyze_product_combinations()