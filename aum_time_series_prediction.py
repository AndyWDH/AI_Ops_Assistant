import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def plot_time_series(series, title='时间序列图'):
    """绘制时间序列图"""
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title(title)
    plt.xlabel('月份')
    plt.ylabel('平均AUM（元）')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300)
    plt.close()

def plot_acf_pacf(series, lags=20):
    """绘制ACF和PACF图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series, ax=ax1, lags=lags)
    ax1.set_title('自相关函数（ACF）')
    
    plot_pacf(series, ax=ax2, lags=lags)
    ax2.set_title('偏自相关函数（PACF）')
    
    plt.tight_layout()
    plt.savefig('acf_pacf.png', dpi=300)
    plt.close()

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
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='训练集')
    plt.plot(test, label='测试集')
    plt.plot(predictions, label='预测值')
    plt.title('ARIMA模型预测结果')
    plt.xlabel('月份')
    plt.ylabel('平均AUM（元）')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300)
    plt.close()
    
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
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='历史数据')
    plt.plot(forecast_df, label='未来预测', linestyle='--', marker='o')
    plt.title('未来季度AUM增长趋势预测')
    plt.xlabel('月份')
    plt.ylabel('平均AUM（元）')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加预测值标签
    for date, value in forecast_df['predicted_aum'].items():
        plt.annotate(f'{value:,.0f}', (date, value), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('future_prediction.png', dpi=300)
    plt.close()
    
    return forecast_df

def analyze_aum_trend():
    """主函数：分析AUM趋势并进行预测"""
    print("开始AUM时间序列分析...")
    
    # 1. 加载并预处理数据
    monthly_aum = load_and_preprocess_data()
    print(f"\n数据加载完成，共 {len(monthly_aum)} 个月的AUM数据")
    print(f"时间范围：{monthly_aum.index.min().strftime('%Y-%m')} 到 {monthly_aum.index.max().strftime('%Y-%m')}")
    
    # 2. 绘制原始时间序列图
    plot_time_series(monthly_aum['total_assets'], 'AUM月度变化趋势')
    
    # 3. 检查平稳性
    is_stationary = check_stationarity(monthly_aum['total_assets'])
    
    # 4. 如果序列不平稳，进行差分
    if not is_stationary:
        diff_series = difference_series(monthly_aum['total_assets'], order=1)
        is_stationary_diff = check_stationarity(diff_series)
        plot_time_series(diff_series, '差分后的AUM序列')
        plot_acf_pacf(diff_series)
    else:
        diff_series = monthly_aum['total_assets']
        plot_acf_pacf(diff_series)
    
    # 5. 训练ARIMA模型
    print("\n训练ARIMA模型...")
    # 默认使用(1,1,1)阶，实际应用中可以根据ACF/PACF图调整
    model_fit = train_arima_model(monthly_aum['total_assets'], order=(1, 1, 1))
    print(model_fit.summary())
    
    # 6. 评估模型
    evaluate_model(model_fit, monthly_aum['total_assets'], test_size=3)
    
    # 7. 预测未来季度
    print("\n预测未来3个月的AUM...")
    forecast_df = predict_future(monthly_aum['total_assets'], model_fit, steps=3)
    
    # 8. 输出预测结果
    print("\n未来3个月AUM预测结果：")
    for date, value in forecast_df['predicted_aum'].items():
        print(f"{date.strftime('%Y-%m')}: {value:,.2f} 元")
    
    # 9. 保存预测结果到CSV
    forecast_df.to_csv('aum_future_prediction.csv')
    print("\n预测结果已保存到 aum_future_prediction.csv")
    
    print("\nAUM时间序列分析完成！")

if __name__ == "__main__":
    analyze_aum_trend()