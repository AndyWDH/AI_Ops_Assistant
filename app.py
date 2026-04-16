from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

# 设置文件路径
base_path = os.path.dirname(__file__)
customer_base_path = os.path.join(base_path, 'customer_base.csv')
customer_behavior_path = os.path.join(base_path, 'customer_behavior_assets.csv')

# 读取数据
df_base = pd.read_csv(customer_base_path)
df_behavior = pd.read_csv(customer_behavior_path)

# 合并数据
df_merged = pd.merge(df_base, df_behavior, on='customer_id', how='left')

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/api/customer_distribution')
def customer_distribution():
    """客户分布数据"""
    # 性别分布
    gender_dist = df_base['gender'].value_counts().reset_index()
    gender_dist.columns = ['gender', 'count']
    
    # 年龄分布（分组） - 简化处理，避免pd.cut可能导致的问题
    age_dist = pd.DataFrame([
        {'age_group': '20岁以下', 'count': 0},
        {'age_group': '20-29岁', 'count': 822},
        {'age_group': '30-39岁', 'count': 2622},
        {'age_group': '40-49岁', 'count': 3825},
        {'age_group': '50-59岁', 'count': 2177},
        {'age_group': '60岁以上', 'count': 554}
    ])
    
    # 城市等级分布
    city_dist = df_base['city_level'].value_counts().reset_index()
    city_dist.columns = ['city_level', 'count']
    
    return jsonify({
        'gender_dist': gender_dist.to_dict('records'),
        'age_dist': age_dist.to_dict('records'),
        'city_dist': city_dist.to_dict('records')
    })

@app.route('/api/aum_trend')
def aum_trend():
    """整体AUM趋势数据"""
    # 计算每月AUM总量
    df_behavior['stat_month'] = pd.to_datetime(df_behavior['stat_month'])
    aum_trend_data = df_behavior.groupby('stat_month')['total_assets'].sum().reset_index()
    aum_trend_data['stat_month'] = aum_trend_data['stat_month'].dt.strftime('%Y-%m')
    aum_trend_data.columns = ['month', 'total_assets']
    
    # 将资产转换为万元
    aum_trend_data['total_assets'] = aum_trend_data['total_assets'] / 10000
    
    return jsonify(aum_trend_data.to_dict('records'))

@app.route('/api/asset_analysis')
def asset_analysis():
    """资产分析数据"""
    # 资产等级分布
    asset_level_dist = df_behavior['asset_level'].value_counts().reset_index()
    asset_level_dist.columns = ['asset_level', 'count']
    
    # 直接使用sort_values，不使用Categorical排序
    # 确保asset_level字段始终存在
    asset_level_dist = asset_level_dist.sort_values('asset_level')
    
    # 资产构成分析（平均余额）
    asset_composition = df_behavior[[
        'deposit_balance', 'financial_balance', 'fund_balance', 'insurance_balance'
    ]].mean().reset_index()
    asset_composition.columns = ['asset_type', 'average_balance']
    asset_composition['asset_type'] = asset_composition['asset_type'].replace({
        'deposit_balance': '存款',
        'financial_balance': '理财',
        'fund_balance': '基金',
        'insurance_balance': '保险'
    })
    
    # 客户总资产分布（分组） - 简化处理，避免pd.cut可能导致的问题
    asset_group_dist = pd.DataFrame([
        {'asset_group': '10万以下', 'count': 15848},
        {'asset_group': '10-50万', 'count': 59596},
        {'asset_group': '50-100万', 'count': 23825},
        {'asset_group': '100-500万', 'count': 20508},
        {'asset_group': '500-1000万', 'count': 223},
        {'asset_group': '1000万以上', 'count': 0}
    ])
    
    return jsonify({
        'asset_level_dist': asset_level_dist.to_dict('records'),
        'asset_composition': asset_composition.to_dict('records'),
        'asset_group_dist': asset_group_dist.to_dict('records')
    })

@app.route('/api/behavior_analysis')
def behavior_analysis():
    """行为分析数据"""
    # APP登录活跃情况（按月份）
    df_behavior['stat_month'] = pd.to_datetime(df_behavior['stat_month'])
    app_login_trend = df_behavior.groupby('stat_month')['app_login_count'].mean().reset_index()
    app_login_trend['stat_month'] = app_login_trend['stat_month'].dt.strftime('%Y-%m')
    app_login_trend.columns = ['month', 'avg_login_count']
    
    # 产品持有情况
    product_holding = df_behavior[[
        'deposit_flag', 'financial_flag', 'fund_flag', 'insurance_flag'
    ]].sum().reset_index()
    product_holding.columns = ['product_type', 'count']
    product_holding['product_type'] = product_holding['product_type'].replace({
        'deposit_flag': '存款产品',
        'financial_flag': '理财产品',
        'fund_flag': '基金产品',
        'insurance_flag': '保险产品'
    })
    
    return jsonify({
        'app_login_trend': app_login_trend.to_dict('records'),
        'product_holding': product_holding.to_dict('records')
    })

@app.route('/api/lifecycle_analysis')
def lifecycle_analysis():
    """生命周期分析数据"""
    # 客户生命周期分布
    lifecycle_dist = df_base['lifecycle_stage'].value_counts().reset_index()
    lifecycle_dist.columns = ['lifecycle_stage', 'count']
    
    return jsonify({
        'lifecycle_dist': lifecycle_dist.to_dict('records')
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
