import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载并预处理数据"""
    # 读取数据
    file_path = 'E:\\AI\\AI大模型应用第17期\\AI大模型应用第17期\\25-项目实战：AI运营助手\\CASE-百万客群经营 20251224\\customer_behavior_assets.csv'
    df = pd.read_csv(file_path)
    
    # 按月份聚合，计算每个月的平均AUM
    monthly_aum = df.groupby('stat_month')['total_assets'].mean().reset_index()
    
    # 将stat_month转换为datetime类型
    monthly_aum['stat_month'] = pd.to_datetime(monthly_aum['stat_month'], format='%Y-%m')
    
    # 按时间排序
    monthly_aum = monthly_aum.sort_values('stat_month')
    
    return monthly_aum

def simple_arima_forecast(data, steps=3):
    """简化版ARIMA预测，使用差分和移动平均"""
    # 提取AUM数据
    aum = data['total_assets'].values
    n = len(aum)
    
    # 1. 计算一阶差分
    diff = np.diff(aum)
    
    # 2. 计算差分的移动平均（MA(1)）
    ma1 = np.mean(diff)
    
    # 3. 计算自回归系数（AR(1)）
    # 使用滞后1期的差分数据
    lag1 = diff[:-1]
    current = diff[1:]
    ar1 = np.corrcoef(lag1, current)[0, 1]
    
    # 4. 预测未来值
    forecasts = []
    last_aum = aum[-1]
    last_diff = diff[-1]
    
    for _ in range(steps):
        # 使用AR(1)和MA(1)预测下一期差分
        next_diff = ar1 * last_diff + ma1
        # 计算下一期AUM
        next_aum = last_aum + next_diff
        # 更新值
        forecasts.append(next_aum)
        last_aum = next_aum
        last_diff = next_diff
    
    return forecasts

def analyze_aum_trend():
    """主函数：分析AUM趋势并进行预测"""
    print("开始AUM时间序列分析...")
    
    # 1. 加载并预处理数据
    monthly_aum = load_and_preprocess_data()
    print(f"\n数据加载完成，共 {len(monthly_aum)} 个月的AUM数据")
    print(f"时间范围：{monthly_aum['stat_month'].min().strftime('%Y-%m')} 到 {monthly_aum['stat_month'].max().strftime('%Y-%m')}")
    
    # 2. 输出历史AUM数据
    print("\n历史AUM数据：")
    for _, row in monthly_aum.iterrows():
        print(f"{row['stat_month'].strftime('%Y-%m')}: {row['total_assets']:,.2f} 元")
    
    # 3. 计算月度增长率
    monthly_aum['growth_rate'] = monthly_aum['total_assets'].pct_change() * 100
    print(f"\n月度增长率统计：")
    print(f"平均月度增长率: {monthly_aum['growth_rate'].mean():.2f}%")
    print(f"最高月度增长率: {monthly_aum['growth_rate'].max():.2f}%")
    print(f"最低月度增长率: {monthly_aum['growth_rate'].min():.2f}%")
    
    # 4. 进行预测
    steps = 3
    forecasts = simple_arima_forecast(monthly_aum, steps=steps)
    
    # 5. 生成未来日期
    last_date = monthly_aum['stat_month'].max()
    future_dates = pd.date_range(start=last_date, periods=steps+1, freq='M')[1:]
    
    # 6. 输出预测结果
    print(f"\n未来{steps}个月AUM预测结果：")
    prediction_data = []
    for date, forecast in zip(future_dates, forecasts):
        print(f"{date.strftime('%Y-%m')}: {forecast:,.2f} 元")
        prediction_data.append({
            'stat_month': date,
            'predicted_aum': forecast
        })
    
    # 7. 保存预测结果到CSV
    prediction_df = pd.DataFrame(prediction_data)
    prediction_df.to_csv('aum_future_prediction.csv', index=False)
    print("\n预测结果已保存到 aum_future_prediction.csv")
    
    # 8. 生成简单分析报告
    generate_simple_report(monthly_aum, prediction_df)
    
    print("\nAUM时间序列分析完成！")

def generate_simple_report(historical_data, prediction_data):
    """生成简单的分析报告"""
    report = "# AUM时间序列分析与预测报告\n\n"
    
    # 1. 数据概览
    report += "## 1. 数据概览\n"
    report += f"- 分析月份数: {len(historical_data)}\n"
    report += f"- 时间范围: {historical_data['stat_month'].min().strftime('%Y-%m')} 至 {historical_data['stat_month'].max().strftime('%Y-%m')}\n"
    report += f"- 平均AUM: {historical_data['total_assets'].mean():,.2f} 元\n"
    report += f"- 最大AUM: {historical_data['total_assets'].max():,.2f} 元\n"
    report += f"- 最小AUM: {historical_data['total_assets'].min():,.2f} 元\n\n"
    
    # 2. 趋势分析
    report += "## 2. AUM历史趋势分析\n"
    report += f"- 平均月度增长率: {historical_data['growth_rate'].mean():.2f}%\n"
    report += f"- 最高月度增长率: {historical_data['growth_rate'].max():.2f}%\n"
    report += f"- 最低月度增长率: {historical_data['growth_rate'].min():.2f}%\n\n"
    
    # 3. 未来预测
    report += "## 3. 未来AUM预测\n"
    report += "基于简化ARIMA模型，预测未来3个月的平均AUM：\n\n"
    report += "| 月份 | 预测AUM（元） |\n"
    report += "|------|---------------|\n"
    
    for _, row in prediction_data.iterrows():
        report += f"| {row['stat_month'].strftime('%Y-%m')} | {row['predicted_aum']:,.2f} |\n"
    
    # 4. 业务建议
    report += "\n## 4. 业务建议\n"
    report += "- 基于预测结果，制定相应的客户营销和产品推广策略\n"
    report += "- 关注AUM增长趋势，及时调整资产配置建议\n"
    report += "- 针对不同客户群体，提供个性化的理财方案\n"
    
    # 保存报告
    with open('aum_simple_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析报告已保存为 aum_simple_analysis_report.md")

if __name__ == "__main__":
    analyze_aum_trend()