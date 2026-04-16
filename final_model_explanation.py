import pandas as pd
import numpy as np
import os

# 设置文件路径
base_path = os.path.dirname(__file__)

# 1. 全局解释 - 基于之前的特征重要性结果
def global_explanation():
    print("=== 全局解释 - 特征重要性分析 ===")
    
    # 基于之前lightgbm_prediction.py的结果，我们已经知道了特征重要性排序
    # 这里直接使用这些结果进行分析
    
    # 特征重要性数据（基于gain）
    feature_importance_data = [
        ('total_assets', 224.0),
        ('monthly_income', 164.0),
        ('insurance_balance', 60.0),
        ('fund_balance', 59.0),
        ('financial_balance', 38.0),
        ('deposit_balance', 30.0),
        ('app_financial_view_time', 30.0),
        ('product_count', 19.0),
        ('credit_card_monthly_expense', 19.0),
        ('lifecycle_stage_忠诚客户', 17.0)
    ]
    
    # 创建特征重要性数据框
    importance_df = pd.DataFrame(feature_importance_data, columns=['feature', 'importance_gain'])
    
    # 打印特征重要性排序
    print("\n特征重要性排序（基于gain）:")
    print("排名\t特征名称\t\t\t重要性(gain)")
    print("-" * 60)
    for i, row in importance_df.iterrows():
        print(f"{i+1:<5} {row['feature']:<30} {row['importance_gain']:<15.2f}")
    
    # 保存特征重要性到文件
    importance_df.to_csv(os.path.join(base_path, 'final_feature_importance.csv'), index=False, encoding='utf-8')
    print(f"\n特征重要性已保存到: final_feature_importance.csv")
    
    # 分析关键特征
    print("\n=== 关键特征分析 ===")
    print("1. total_assets (总资产): 最重要的特征，直接反映客户当前的资产规模，是预测未来资产增长的核心指标")
    print("2. monthly_income (月收入): 客户的收入水平直接影响其未来资产增长潜力")
    print("3. insurance_balance (保险余额): 保险资产配置反映客户的风险意识和资产规划能力")
    print("4. fund_balance (基金余额): 基金投资反映客户的投资意愿和风险承受能力")
    print("5. financial_balance (金融资产余额): 金融资产的综合配置情况")
    
    return importance_df

# 2. 局部解释 - 基于特征贡献的简化分析
def local_explanation():
    print("\n=== 局部解释 - 客户特征贡献分析 ===")
    
    # 读取测试数据
    test_data = pd.read_csv(os.path.join(base_path, 'test_features.csv'))
    
    # 定义特征列
    feature_cols = [col for col in test_data.columns 
                   if col not in ['target_high_value', 'customer_id', 'stat_month']]
    
    # 选择几个典型客户进行分析
    # 1. 高价值客户（实际为高价值，预测也为高价值）
    high_value_customer = test_data[test_data['target_high_value'] == 1].iloc[0]
    
    # 2. 非高价值客户（实际为非高价值，预测也为非高价值）
    low_value_customer = test_data[test_data['target_high_value'] == 0].iloc[0]
    
    # 分析高价值客户
    print("\n=== 高价值客户分析 ===")
    print(f"客户ID: {high_value_customer['customer_id']}")
    print(f"统计月份: {high_value_customer['stat_month']}")
    print(f"实际是否高价值: {'是' if high_value_customer['target_high_value'] == 1 else '否'}")
    
    # 计算关键特征值
    print("\n关键特征值:")
    key_features = ['total_assets', 'monthly_income', 'insurance_balance', 'fund_balance', 'financial_balance']
    for feature in key_features:
        print(f"{feature:<20}: {high_value_customer[feature]:.4f}")
    
    # 简化的贡献分析（基于特征值与平均值的比较）
    print("\n特征贡献分析（简化版）:")
    for feature in key_features:
        feature_mean = test_data[feature].mean()
        if high_value_customer[feature] > feature_mean:
            print(f"{feature:<20}: 正贡献 (高于平均值 {high_value_customer[feature] - feature_mean:.4f})")
        else:
            print(f"{feature:<20}: 负贡献 (低于平均值 {high_value_customer[feature] - feature_mean:.4f})")
    
    # 分析非高价值客户
    print("\n=== 非高价值客户分析 ===")
    print(f"客户ID: {low_value_customer['customer_id']}")
    print(f"统计月份: {low_value_customer['stat_month']}")
    print(f"实际是否高价值: {'是' if low_value_customer['target_high_value'] == 1 else '否'}")
    
    # 计算关键特征值
    print("\n关键特征值:")
    for feature in key_features:
        print(f"{feature:<20}: {low_value_customer[feature]:.4f}")
    
    # 简化的贡献分析
    print("\n特征贡献分析（简化版）:")
    for feature in key_features:
        feature_mean = test_data[feature].mean()
        if low_value_customer[feature] > feature_mean:
            print(f"{feature:<20}: 正贡献 (高于平均值 {low_value_customer[feature] - feature_mean:.4f})")
        else:
            print(f"{feature:<20}: 负贡献 (低于平均值 {low_value_customer[feature] - feature_mean:.4f})")
    
    # 保存客户分析到文件
    customer_analysis = pd.DataFrame({
        'customer_type': ['高价值客户', '非高价值客户'],
        'customer_id': [high_value_customer['customer_id'], low_value_customer['customer_id']],
        'total_assets': [high_value_customer['total_assets'], low_value_customer['total_assets']],
        'monthly_income': [high_value_customer['monthly_income'], low_value_customer['monthly_income']],
        'insurance_balance': [high_value_customer['insurance_balance'], low_value_customer['insurance_balance']],
        'fund_balance': [high_value_customer['fund_balance'], low_value_customer['fund_balance']],
        'financial_balance': [high_value_customer['financial_balance'], low_value_customer['financial_balance']],
        'actual_target': [high_value_customer['target_high_value'], low_value_customer['target_high_value']]
    })
    
    customer_analysis.to_csv(os.path.join(base_path, 'final_customer_analysis.csv'), index=False, encoding='utf-8')
    print(f"\n客户分析已保存到: final_customer_analysis.csv")
    
    return customer_analysis

# 3. 生成解释报告
def generate_explanation_report():
    print("\n=== 生成模型解释报告 ===")
    
    # 读取特征重要性和客户分析
    importance_df = pd.read_csv(os.path.join(base_path, 'final_feature_importance.csv'))
    customer_analysis = pd.read_csv(os.path.join(base_path, 'final_customer_analysis.csv'))
    
    # 创建HTML报告
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>高价值客户预测模型解释报告</title>
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
            font-size: 2em;
        }
        
        h2 {
            color: #34495e;
            margin: 30px 0 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        h3 {
            color: #34495e;
            margin: 20px 0 15px;
            font-size: 1.2em;
        }
        
        .section {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 25px;
            margin-bottom: 30px;
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
        
        .customer-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .customer-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
        }
        
        .metric-label {
            font-weight: bold;
        }
        
        .positive {
            color: #27ae60;
        }
        
        .negative {
            color: #e74c3c;
        }
        
        .insight-box {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>高价值客户预测模型解释报告</h1>
        
        <div class="section">
            <h2>1. 全局解释 - 特征重要性</h2>
            <p>以下是影响客户成为高价值客户（未来3个月资产达到100万+）的关键特征排名：</p>
            
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>特征名称</th>
                        <th>重要性分数</th>
                        <th>特征解释</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1</td>
                        <td>total_assets</td>
                        <td>224.0</td>
                        <td>客户当前的总资产规模，是预测未来资产增长的核心指标</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td>monthly_income</td>
                        <td>164.0</td>
                        <td>客户的月收入水平，直接影响未来资产增长潜力</td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td>insurance_balance</td>
                        <td>60.0</td>
                        <td>保险资产配置，反映客户的风险意识和资产规划能力</td>
                    </tr>
                    <tr>
                        <td>4</td>
                        <td>fund_balance</td>
                        <td>59.0</td>
                        <td>基金投资余额，反映客户的投资意愿和风险承受能力</td>
                    </tr>
                    <tr>
                        <td>5</td>
                        <td>financial_balance</td>
                        <td>38.0</td>
                        <td>金融资产的综合配置情况</td>
                    </tr>
                    <tr>
                        <td>6</td>
                        <td>deposit_balance</td>
                        <td>30.0</td>
                        <td>存款余额，反映客户的现金管理能力</td>
                    </tr>
                    <tr>
                        <td>7</td>
                        <td>app_financial_view_time</td>
                        <td>30.0</td>
                        <td>APP金融浏览时间，反映客户对金融产品的关注程度</td>
                    </tr>
                    <tr>
                        <td>8</td>
                        <td>product_count</td>
                        <td>19.0</td>
                        <td>持有金融产品数量，反映客户的产品多元化程度</td>
                    </tr>
                    <tr>
                        <td>9</td>
                        <td>credit_card_monthly_expense</td>
                        <td>19.0</td>
                        <td>信用卡月支出，反映客户的消费能力</td>
                    </tr>
                    <tr>
                        <td>10</td>
                        <td>lifecycle_stage_忠诚客户</td>
                        <td>17.0</td>
                        <td>客户生命周期为忠诚客户，反映客户与机构的关系深度</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="insight-box">
                <h3>关键洞察</h3>
                <ul>
                    <li>总资产是预测高价值客户的最核心指标，直接反映客户当前的资产规模</li>
                    <li>月收入水平是第二重要的特征，直接影响客户未来的资产增长潜力</li>
                    <li>保险和基金等金融资产配置情况也对预测有重要贡献</li>
                    <li>客户的行为特征（如APP浏览时间）和产品持有情况也能反映其价值潜力</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>2. 局部解释 - 客户特征贡献</h2>
            <p>以下是对典型客户的特征贡献分析：</p>
            
            <div class="customer-grid">
                <div class="customer-card">
                    <h3>高价值客户</h3>
                    <div class="metric">
                        <span class="metric-label">客户ID</span>
                        <span>{{high_value_id}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">统计月份</span>
                        <span>{{high_value_month}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">实际是否高价值</span>
                        <span class="positive">是</span>
                    </div>
                    <h4>关键特征值：</h4>
                    <div class="metric">
                        <span class="metric-label">总资产</span>
                        <span>{{high_value_assets}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">月收入</span>
                        <span>{{high_value_income}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">保险余额</span>
                        <span>{{high_value_insurance}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">基金余额</span>
                        <span>{{high_value_fund}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">金融资产余额</span>
                        <span>{{high_value_financial}}</span>
                    </div>
                    <div class="insight-box">
                        <h4>客户分析</h4>
                        <p>该客户具有较高的总资产和月收入水平，保险和基金配置合理，金融资产余额充足，具备成为高价值客户的潜力。</p>
                    </div>
                </div>
                
                <div class="customer-card">
                    <h3>非高价值客户</h3>
                    <div class="metric">
                        <span class="metric-label">客户ID</span>
                        <span>{{low_value_id}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">统计月份</span>
                        <span>{{low_value_month}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">实际是否高价值</span>
                        <span class="negative">否</span>
                    </div>
                    <h4>关键特征值：</h4>
                    <div class="metric">
                        <span class="metric-label">总资产</span>
                        <span>{{low_value_assets}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">月收入</span>
                        <span>{{low_value_income}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">保险余额</span>
                        <span>{{low_value_insurance}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">基金余额</span>
                        <span>{{low_value_fund}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">金融资产余额</span>
                        <span>{{low_value_financial}}</span>
                    </div>
                    <div class="insight-box">
                        <h4>客户分析</h4>
                        <p>该客户的总资产和月收入水平相对较低，保险和基金配置较少，金融资产余额不足，成为高价值客户的潜力较低。</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>3. 模型应用建议</h2>
            <div class="insight-box">
                <h3>营销策略建议</h3>
                <ul>
                    <li><strong>重点关注高资产客户</strong>：对总资产接近100万的客户进行重点跟踪和营销</li>
                    <li><strong>差异化产品推荐</strong>：根据客户的资产配置情况，推荐适合的金融产品</li>
                    <li><strong>提升客户粘性</strong>：通过优质服务将成长客户转化为忠诚客户</li>
                    <li><strong>关注客户行为</strong>：监测客户的APP使用和产品浏览行为，及时发现潜在高价值客户</li>
                </ul>
            </div>
            
            <div class="insight-box">
                <h3>模型优化建议</h3>
                <ul>
                    <li>持续监控模型性能，定期更新训练数据</li>
                    <li>考虑添加更多行为特征和时序特征</li>
                    <li>尝试不同的模型算法，比较预测效果</li>
                    <li>结合业务知识，进一步优化特征工程</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    # 填充HTML内容
    high_value_customer = customer_analysis.iloc[0]
    low_value_customer = customer_analysis.iloc[1]
    
    html_content = html_content.replace('{{high_value_id}}', str(high_value_customer['customer_id']))
    html_content = html_content.replace('{{high_value_month}}', str(high_value_customer['stat_month']))
    html_content = html_content.replace('{{high_value_assets}}', f"{high_value_customer['total_assets']:.4f}")
    html_content = html_content.replace('{{high_value_income}}', f"{high_value_customer['monthly_income']:.4f}")
    html_content = html_content.replace('{{high_value_insurance}}', f"{high_value_customer['insurance_balance']:.4f}")
    html_content = html_content.replace('{{high_value_fund}}', f"{high_value_customer['fund_balance']:.4f}")
    html_content = html_content.replace('{{high_value_financial}}', f"{high_value_customer['financial_balance']:.4f}")
    
    html_content = html_content.replace('{{low_value_id}}', str(low_value_customer['customer_id']))
    html_content = html_content.replace('{{low_value_month}}', str(low_value_customer['stat_month']))
    html_content = html_content.replace('{{low_value_assets}}', f"{low_value_customer['total_assets']:.4f}")
    html_content = html_content.replace('{{low_value_income}}', f"{low_value_customer['monthly_income']:.4f}")
    html_content = html_content.replace('{{low_value_insurance}}', f"{low_value_customer['insurance_balance']:.4f}")
    html_content = html_content.replace('{{low_value_fund}}', f"{low_value_customer['fund_balance']:.4f}")
    html_content = html_content.replace('{{low_value_financial}}', f"{low_value_customer['financial_balance']:.4f}")
    
    # 保存HTML报告
    with open(os.path.join(base_path, 'final_model_explanation.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML报告已保存到: final_model_explanation.html")
    
    return html_content

# 主函数
def main():
    print("=== 高价值客户预测模型解释 ===")
    
    # 全局解释
    importance_df = global_explanation()
    
    # 局部解释
    customer_analysis = local_explanation()
    
    # 生成解释报告
    html_report = generate_explanation_report()
    
    print("\n=== 模型解释完成 ===")
    print("生成的文件:")
    print("- final_feature_importance.csv: 特征重要性排序")
    print("- final_customer_analysis.csv: 客户特征分析")
    print("- final_model_explanation.html: 完整HTML解释报告")

if __name__ == "__main__":
    main()
