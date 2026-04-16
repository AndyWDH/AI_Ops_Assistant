import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
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
    
    return X_train, X_test, y_train, y_test, feature_cols

# 训练决策树模型
def train_decision_tree(X_train, y_train, max_depth=4):
    print(f"\n训练决策树模型 (max_depth={max_depth})...")
    
    # 创建决策树分类器
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        criterion='gini'
    )
    
    # 训练模型
    clf.fit(X_train, y_train)
    
    print("模型训练完成")
    return clf

# 模型评估
def evaluate_model(clf, X_test, y_test):
    print("\n评估模型性能...")
    
    # 预测
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_pred_proba

# 生成文本形式的决策树
def generate_tree_text(clf, feature_cols):
    print("\n生成文本形式的决策树...")
    
    tree_text = export_text(
        clf,
        feature_names=feature_cols,
        max_depth=4,
        decimals=3,
        show_weights=True
    )
    
    print("决策树文本:")
    print(tree_text)
    
    # 保存到文件
    tree_text_path = os.path.join(base_path, 'decision_tree_text.txt')
    with open(tree_text_path, 'w', encoding='utf-8') as f:
        f.write(tree_text)
    print(f"\n决策树文本已保存到: {tree_text_path}")
    
    return tree_text

# 生成决策树可视化图片
def plot_decision_tree(clf, feature_cols):
    print("\n生成决策树可视化图片...")
    
    # 创建可视化
    plt.figure(figsize=(20, 15))
    
    plot_tree(
        clf,
        feature_names=feature_cols,
        class_names=['非高价值客户', '高价值客户'],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=4,
        impurity=True,
        proportion=True,
        precision=3
    )
    
    plt.title(f"决策树可视化 (max_depth={clf.max_depth})")
    plt.tight_layout()
    
    # 保存图片
    tree_plot_path = os.path.join(base_path, 'decision_tree_plot.png')
    plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"决策树可视化已保存到: {tree_plot_path}")
    return tree_plot_path

# 生成Mermaid格式的决策树
def generate_mermaid_tree(clf, feature_cols):
    print("\n生成Mermaid格式的决策树...")
    
    # 获取树的结构
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value
    impurity = clf.tree_.impurity
    
    # 创建Mermaid图
    mermaid_code = "graph TD\n    "
    
    # 遍历所有节点
    for i in range(n_nodes):
        # 如果是叶节点
        if children_left[i] == children_right[i]:
            # 计算类别概率
            class_probs = value[i][0] / value[i][0].sum()
            class_0_prob = class_probs[0]
            class_1_prob = class_probs[1]
            
            # 确定最终类别
            final_class = "高价值客户" if class_1_prob > class_0_prob else "非高价值客户"
            class_color = "#d4edda" if final_class == "高价值客户" else "#f8d7da"
            
            # 添加叶节点
            mermaid_code += f"    node{i}[叶节点 {i}<br/>类别: {final_class}<br/>非高价值概率: {class_0_prob:.3f}<br/>高价值概率: {class_1_prob:.3f}]:::{final_class.replace(' ', '')}Class\n    "
        else:
            # 内部节点
            feature_name = feature_cols[feature[i]]
            threshold_val = threshold[i]
            
            # 添加内部节点
            mermaid_code += f"    node{i}[节点 {i}<br/>{feature_name} ≤ {threshold_val:.3f}?<br/>基尼系数: {impurity[i]:.3f}]\n    "
            
            # 添加左分支（<= threshold）
            mermaid_code += f"    node{i} -->|是| node{children_left[i]}\n    "
            
            # 添加右分支（> threshold）
            mermaid_code += f"    node{i} -->|否| node{children_right[i]}\n    "
    
    # 添加样式
    mermaid_code += "\n    classDef 高价值客户Class fill:#d4edda,stroke:#c3e6cb,stroke-width:2px;\n    "
    mermaid_code += "    classDef 非高价值客户Class fill:#f8d7da,stroke:#f5c6cb,stroke-width:2px;\n"
    
    # 保存Mermaid代码到文件
    mermaid_path = os.path.join(base_path, 'decision_tree_mermaid.txt')
    with open(mermaid_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    
    print(f"Mermaid代码已保存到: {mermaid_path}")
    return mermaid_code

# 创建HTML报告
def create_html_report(clf, feature_cols, tree_text, mermaid_code, tree_plot_path, df_train, df_test):
    print("\n创建HTML报告...")
    
    # 计算特征重要性
    feature_importance = clf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 获取前10个重要特征
    top_features = importance_df.head(10)
    
    # 生成SVG格式的特征重要性图
    svg_chart = generate_svg_importance(top_features)
    
    # 创建HTML内容
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>潜在高价值客户预测 - 决策树分析报告</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        
        h2 {
            color: #34495e;
            margin: 30px 0 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        h3 {
            color: #34495e;
            margin: 20px 0 15px;
            font-size: 1.4em;
        }
        
        .section {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        
        .metric-label {
            font-size: 1.1em;
            color: #666;
            margin-top: 10px;
        }
        
        .feature-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .feature-table th, .feature-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        .feature-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            color: #333;
        }
        
        .feature-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .tree-text {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-x: auto;
            margin: 20px 0;
        }
        
        .mermaid-container {
            background-color: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            overflow-x: auto;
        }
        
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        
        .chart-image {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        .svg-container {
            width: 100%;
            height: 400px;
            margin: 20px 0;
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }
        
        .insight-box {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        
        ul {
            margin-left: 20px;
            margin-bottom: 20px;
        }
        
        li {
            margin-bottom: 10px;
        }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.4.0/dist/mermaid.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>潜在高价值客户预测 - 决策树分析报告</h1>
        
        <div class="section">
            <h2>1. 分析概述</h2>
            <p>本报告使用决策树模型（max_depth=4）预测客户未来3个月是否能成为高价值客户（资产达到100万+）。</p>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">20,000</div>
                    <div class="metric-label">训练样本数量</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">20,000</div>
                    <div class="metric-label">测试样本数量</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">41</div>
                    <div class="metric-label">特征数量</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">4</div>
                    <div class="metric-label">决策树深度</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>2. 决策树模型分析</h2>
            
            <h3>2.1 决策树可视化</h3>
            <div class="chart-container">
                <img src="decision_tree_plot.png" alt="决策树可视化" class="chart-image">
            </div>
            
            <h3>2.2 Mermaid交互式决策树</h3>
            <div class="mermaid-container">
                <div class="mermaid">
                ''' + mermaid_code + '''
                </div>
            </div>
            
            <h3>2.3 决策树文本表示</h3>
            <div class="tree-text">
            ''' + tree_text + '''
            </div>
        </div>
        
        <div class="section">
            <h2>3. 特征重要性分析</h2>
            
            <h3>3.1 特征重要性排名</h3>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>特征名称</th>
                        <th>重要性分数</th>
                    </tr>
                </thead>
                <tbody>
                ''' + ''.join([f"<tr><td>{i+1}</td><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>" 
                             for i, row in top_features.iterrows()]) + '''
                </tbody>
            </table>
            
            <h3>3.2 特征重要性可视化</h3>
            <div class="svg-container">
            ''' + svg_chart + '''
            </div>
        </div>
        
        <div class="section">
            <h2>4. 潜在高价值客户画像</h2>
            
            <h3>4.1 客户特征概览</h3>
            <div class="mermaid-container">
                <div class="mermaid">
                    graph TD
                        A[潜在高价值客户画像] --> B[资产特征]
                        A --> C[人口统计特征]
                        A --> D[行为特征]
                        A --> E[生命周期特征]
                        
                        B --> B1[总资产较高]
                        B --> B2[金融资产余额高]
                        B --> B3[存款余额稳定]
                        B --> B4[保险和基金配置合理]
                        
                        C --> C1[月收入水平高]
                        C --> C2[职业为企业高管/私营业主]
                        C --> C3[年龄在30-50岁]
                        C --> C4[已婚]
                        
                        D --> D1[投资行为活跃]
                        D --> D2[产品持有种类多]
                        D --> D3[APP使用频繁]
                        
                        E --> E1[生命周期为忠诚客户]
                        E --> E2[非成长客户]
                    
                    classDef positive fill:#d4edda,stroke:#c3e6cb,stroke-width:2px;
                    classDef negative fill:#f8d7da,stroke:#f5c6cb,stroke-width:2px;
                    
                    class B1,B2,B3,B4,C1,C2,C3,C4,D1,D2,D3,E1 positive;
                    class E2 negative;
                </div>
            </div>
            
            <h3>4.2 决策树规则解读</h3>
            <div class="insight-box">
                <h4>高价值客户识别规则：</h4>
                <ul>
                    <li>当客户总资产较高时，更可能成为高价值客户</li>
                    <li>金融资产余额是重要的预测指标</li>
                    <li>职业类型（如企业高管、私营业主）对预测结果有显著影响</li>
                    <li>客户生命周期阶段（如忠诚客户）是重要的参考因素</li>
                    <li>投资行为活跃度反映了客户的资产增长潜力</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>5. 营销建议</h2>
            
            <ul>
                <li><strong>重点关注高资产客户</strong>：对总资产接近100万的客户进行重点跟踪和营销</li>
                <li><strong>差异化营销策略</strong>：根据客户的特征制定个性化的营销方案</li>
                <li><strong>提升客户粘性</strong>：通过优质服务将成长客户转化为忠诚客户</li>
                <li><strong>优化产品推荐</strong>：根据客户的资产配置情况推荐合适的金融产品</li>
                <li><strong>定期监测客户行为</strong>：跟踪客户的投资行为变化，及时调整营销策略</li>
            </ul>
        </div>
    </div>
    
    <script>
        // 初始化Mermaid
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            flowchart: {
                curve: 'linear'
            }
        });
    </script>
</body>
</html>'''
    
    # 保存HTML报告
    html_path = os.path.join(base_path, '高价值客户决策树分析.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已保存到: {html_path}")
    return html_path

# 生成SVG格式的特征重要性图
def generate_svg_importance(importance_df):
    # 准备数据
    features = importance_df['feature'].tolist()
    importances = importance_df['importance'].tolist()
    
    # 设置SVG尺寸
    width = 1000
    height = 400
    margin = 50
    
    # 计算最大高度
    max_importance = max(importances)
    
    # 计算条形宽度和间距
    bar_width = (width - 2 * margin) / len(features) * 0.8
    bar_spacing = (width - 2 * margin) / len(features) * 0.2
    
    # 生成SVG
    svg = f'<svg width="100%" height="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    
    # 添加标题
    svg += f'<text x="{width/2}" y="30" font-size="20" text-anchor="middle" font-weight="bold">特征重要性排名</text>\n'
    
    # 添加x轴
    svg += f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="2"/>\n'
    
    # 添加y轴
    svg += f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" stroke-width="2"/>\n'
    
    # 绘制条形图
    for i, (feature, importance) in enumerate(zip(features, importances)):
        # 计算位置
        x = margin + i * (bar_width + bar_spacing)
        bar_height = (importance / max_importance) * (height - 2 * margin)
        y = height - margin - bar_height
        
        # 绘制条形
        svg += f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#3498db" rx="3" ry="3"/>\n'
        
        # 添加特征名称
        svg += f'<text x="{x + bar_width/2}" y="{height - margin + 20}" font-size="12" text-anchor="middle" fill="#666">{feature}</text>\n'
        
        # 添加重要性值
        svg += f'<text x="{x + bar_width/2}" y="{y - 5}" font-size="10" text-anchor="middle" font-weight="bold">{importance:.4f}</text>\n'
    
    svg += '</svg>'
    return svg

# 主函数
def main():
    print("=== 潜在高价值客户预测 - 决策树分析 ===")
    
    # 加载数据
    df_train, df_test = load_data()
    
    # 准备特征和目标变量
    X_train, X_test, y_train, y_test, feature_cols = prepare_features(df_train, df_test)
    
    # 训练决策树模型
    clf = train_decision_tree(X_train, y_train, max_depth=4)
    
    # 评估模型
    y_pred, y_pred_proba = evaluate_model(clf, X_test, y_test)
    
    # 生成文本形式的决策树
    tree_text = generate_tree_text(clf, feature_cols)
    
    # 生成决策树可视化
    tree_plot_path = plot_decision_tree(clf, feature_cols)
    
    # 生成Mermaid格式的决策树
    mermaid_code = generate_mermaid_tree(clf, feature_cols)
    
    # 创建HTML报告
    html_path = create_html_report(clf, feature_cols, tree_text, mermaid_code, tree_plot_path, df_train, df_test)
    
    print("\n=== 分析完成 ===")
    print(f"生成的文件:")
    print(f"- 决策树文本: {os.path.join(base_path, 'decision_tree_text.txt')}")
    print(f"- 决策树可视化: {tree_plot_path}")
    print(f"- Mermaid代码: {os.path.join(base_path, 'decision_tree_mermaid.txt')}")
    print(f"- HTML报告: {html_path}")

if __name__ == "__main__":
    main()
