import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
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
    
    # 设置时间索引
    monthly_aum.set_index('stat_month', inplace=True)
    
    return monthly_aum

def check_stationarity(series):
    """检查时间序列的平稳性"""
    result = adfuller(series)
    print('ADF统计量:', result[0])
    print('p值:', result[1])
    print('临界值:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    if result[1] <= 0.05:
        print('\n结论：序列是平稳的（拒绝原假设）')
        return True
    else:
        print('\n结论：序列是非平稳的（接受原假设）')
        return False

def difference_series(series, order=1):
    """对序列进行差分"""
    diff = series.copy()
    for i in range(order):
        diff = diff.diff().dropna()
    return diff

def train_arima_model(series, order=(1, 1, 1)):
    """训练ARIMA模型"""
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def evaluate_model(model_fit, series, test_size=3):
    """评估模型性能"""
    # 分割训练集和测试集
    train = series[:-test_size]
    test = series[-test_size:]
    
    # 重新训练模型
    model = ARIMA(train, order=model_fit.model_order)
    model_fit = model.fit()
    
    # 预测
    predictions = model_fit.forecast(steps=test_size)
    
    # 计算RMSE
    rmse = sqrt(mean_squared_error(test, predictions))
    print(f'\n模型评估结果：')
    print(f'Test RMSE: {rmse:.2f}')
    
    # 输出预测与实际值对比
    print('\n预测与实际值对比：')
    for actual, predicted in zip(test, predictions):
        print(f'实际值: {actual:,.2f}, 预测值: {predicted:,.2f}, 误差: {abs(actual-predicted):,.2f}')
    
    return rmse

def predict_future(series, model_fit, steps=3):
    """预测未来季度AUM"""
    # 预测未来steps个月
    future_forecast = model_fit.forecast(steps=steps)
    
    # 生成未来日期索引
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=steps+1, freq='M')[1:]
    
    # 创建预测结果DataFrame
    forecast_df = pd.DataFrame({
        'stat_month': future_dates,
        'predicted_aum': future_forecast
    })
    
    forecast_df.set_index('stat_month', inplace=True)
    
    return forecast_df

def analyze_aum_trend():
    """主函数：分析AUM趋势并进行预测"""
    print("开始AUM时间序列分析...")
    
    # 1. 加载并预处理数据
    monthly_aum = load_and_preprocess_data()
    print(f"\n数据加载完成，共 {len(monthly_aum)} 个月的AUM数据")
    print(f"时间范围：{monthly_aum.index.min().strftime('%Y-%m')} 到 {monthly_aum.index.max().strftime('%Y-%m')}")
    
    # 2. 检查平稳性
    is_stationary = check_stationarity(monthly_aum['total_assets'])
    
    # 3. 如果序列不平稳，进行差分
    if not is_stationary:
        diff_series = difference_series(monthly_aum['total_assets'], order=1)
        is_stationary_diff = check_stationarity(diff_series)
        print(f"\n差分后序列长度: {len(diff_series)}")
    
    # 4. 训练ARIMA模型
    print("\n训练ARIMA模型...")
    # 默认使用(1,1,1)阶，实际应用中可以根据ACF/PACF图调整
    model_fit = train_arima_model(monthly_aum['total_assets'], order=(1, 1, 1))
    print(model_fit.summary())
    
    # 5. 评估模型
    evaluate_model(model_fit, monthly_aum['total_assets'], test_size=3)
    
    # 6. 预测未来季度
    print("\n预测未来3个月的AUM...")
    forecast_df = predict_future(monthly_aum['total_assets'], model_fit, steps=3)
    
    # 7. 输出预测结果
    print("\n未来3个月AUM预测结果：")
    for date, value in forecast_df['predicted_aum'].items():
        print(f"{date.strftime('%Y-%m')}: {value:,.2f} 元")
    
    # 8. 保存预测结果到CSV
    forecast_df.to_csv('aum_future_prediction.csv')
    print("\n预测结果已保存到 aum_future_prediction.csv")
    
    # 9. 生成分析报告
    generate_analysis_report(monthly_aum, forecast_df)
    
    print("\nAUM时间序列分析完成！")

def generate_analysis_report(monthly_aum, forecast_df):
    """生成AUM趋势分析报告"""
    report = "# AUM时间序列分析与预测报告\n\n"
    
    # 1. 数据概览
    report += "## 1. 数据概览\n"
    report += f"- 分析月份数: {len(monthly_aum)}\n"
    report += f"- 时间范围: {monthly_aum.index.min().strftime('%Y-%m')} 至 {monthly_aum.index.max().strftime('%Y-%m')}\n"
    report += f"- 平均AUM: {monthly_aum['total_assets'].mean():,.2f} 元\n"
    report += f"- 最大AUM: {monthly_aum['total_assets'].max():,.2f} 元\n"
    report += f"- 最小AUM: {monthly_aum['total_assets'].min():,.2f} 元\n\n"
    
    # 2. 趋势分析
    report += "## 2. AUM历史趋势分析\n"
    
    # 计算月度增长率
    monthly_aum['growth_rate'] = monthly_aum['total_assets'].pct_change() * 100
    
    report += "- 月度增长率统计:\n"
    report += f"  * 平均月度增长率: {monthly_aum['growth_rate'].mean():.2f}%\n"
    report += f"  * 最高月度增长率: {monthly_aum['growth_rate'].max():.2f}%\n"
    report += f"  * 最低月度增长率: {monthly_aum['growth_rate'].min():.2f}%\n\n"
    
    # 3. ARIMA模型结果
    report += "## 3. ARIMA模型结果\n"
    report += "使用ARIMA(1,1,1)模型进行预测，模型基本信息：\n"
    report += "- 模型类型: ARIMA（自回归积分滑动平均模型）\n"
    report += "- 模型参数: p=1, d=1, q=1\n"
    report += "  * p=1: 自回归项数量\n"
    report += "  * d=1: 差分阶数\n"
    report += "  * q=1: 移动平均项数量\n\n"
    
    # 4. 未来预测
    report += "## 4. 未来AUM预测\n"
    report += "基于ARIMA模型，预测未来3个月的平均AUM：\n\n"
    report += "| 月份 | 预测AUM（元） | 环比增长（元） | 环比增长率 |\n"
    report += "|------|---------------|----------------|------------|\n"
    
    # 计算环比增长
    last_actual = monthly_aum['total_assets'].iloc[-1]
    previous_value = last_actual
    
    for date, row in forecast_df.iterrows():
        predicted = row['predicted_aum']
        growth = predicted - previous_value
        growth_rate = (growth / previous_value) * 100
        report += f"| {date.strftime('%Y-%m')} | {predicted:,.2f} | {growth:,.2f} | {growth_rate:.2f}% |\n"
        previous_value = predicted
    
    # 5. 业务建议
    report += "\n## 5. 业务建议\n"
    
    # 计算整体预测趋势
    total_growth = forecast_df['predicted_aum'].iloc[-1] - last_actual
    total_growth_rate = (total_growth / last_actual) * 100
    
    report += f"基于未来3个月的预测，预计AUM将{'增长' if total_growth > 0 else '下降'} {abs(total_growth):,.2f}元（{'+' if total_growth > 0 else ''}{total_growth_rate:.2f}%）。\n\n"
    
    report += "### 营销策略建议：\n"
    if total_growth > 0:
        report += "- 维持现有营销力度，重点关注高增长客户群体\n"
        report += "- 推出季度理财产品，吸引客户增加资产配置\n"
        report += "- 针对不同客户群体制定差异化的资产增值方案\n"
    else:
        report += "- 分析AUM下降原因，制定挽留策略\n"
        report += "- 推出短期高收益产品，刺激客户资产增长\n"
        report += "- 加强客户沟通，了解客户需求变化\n"
    
    report += "\n### 产品策略建议：\n"
    report += "- 根据AUM增长趋势，调整产品组合结构\n"
    report += "- 重点推广与AUM增长相关性高的产品\n"
    report += "- 开发创新金融产品，满足客户多元化需求\n"
    
    # 保存报告
    with open('aum_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析报告已保存为 aum_analysis_report.md")

if __name__ == "__main__":
    analyze_aum_trend()