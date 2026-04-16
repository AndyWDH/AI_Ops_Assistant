import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# 设置文件路径
base_path = os.path.dirname(__file__)
customer_base_path = os.path.join(base_path, 'customer_base.csv')
customer_behavior_path = os.path.join(base_path, 'customer_behavior_assets.csv')
output_path = os.path.join(base_path, 'processed_features.csv')

# 读取数据
df_base = pd.read_csv(customer_base_path)
df_behavior = pd.read_csv(customer_behavior_path)

# 数据预处理
def preprocess_data():
    # 合并数据
    df_merged = pd.merge(df_base, df_behavior, on='customer_id', how='left')
    
    # 将stat_month转换为日期类型
    df_merged['stat_month'] = pd.to_datetime(df_merged['stat_month'])
    
    # 提取月份和季度特征
    df_merged['month'] = df_merged['stat_month'].dt.month
    df_merged['quarter'] = df_merged['stat_month'].dt.quarter
    
    return df_merged

# 静态特征处理
def process_static_features(df):
    # 选择静态特征
    static_features = ['age', 'occupation_type', 'monthly_income', 'city_level', 'marriage_status', 'lifecycle_stage']
    
    # 缺失值处理
    df[static_features] = df[static_features].fillna({
        'age': df['age'].median(),
        'occupation_type': '未知',
        'monthly_income': df['monthly_income'].median(),
        'city_level': '未知',
        'marriage_status': '未知',
        'lifecycle_stage': '未知'
    })
    
    return df[static_features]

# 动态特征处理
def process_dynamic_features(df):
    print("开始处理动态特征...")
    # 选择动态特征
    dynamic_features = [
        'total_assets', 'deposit_balance', 'financial_balance', 'fund_balance', 
        'insurance_balance', 'product_count', 'investment_monthly_count', 
        'app_login_count', 'app_financial_view_time', 'financial_repurchase_count', 
        'credit_card_monthly_expense'
    ]
    
    print(f"动态特征列表：{dynamic_features}")
    
    # 缺失值处理
    print("处理缺失值...")
    df[dynamic_features] = df[dynamic_features].fillna(0)
    
    # 计算资产增长率（与上月相比）
    print("计算资产增长率...")
    df = df.sort_values(['customer_id', 'stat_month'])
    df['asset_growth_rate'] = df.groupby('customer_id')['total_assets'].pct_change()
    df['asset_growth_rate'] = df['asset_growth_rate'].fillna(0)
    
    # 计算环比变化
    print("计算环比变化...")
    df['asset_mom_change'] = df.groupby('customer_id')['total_assets'].diff()
    df['asset_mom_change'] = df['asset_mom_change'].fillna(0)
    
    # 计算累计增长额
    print("计算累计增长额...")
    df['asset_cumulative_growth'] = df.groupby('customer_id')['asset_mom_change'].cumsum()
    
    # 简化：只计算少量关键滚动特征，避免处理大量数据
    print("计算关键滚动特征...")
    key_features = ['total_assets', 'financial_balance', 'fund_balance']
    roll_window = 3
    for feature in key_features:
        print(f"处理特征：{feature}")
        df[f'{feature}_rolling_mean'] = df.groupby('customer_id')[feature].rolling(roll_window).mean().reset_index(level=0, drop=True)
        df[f'{feature}_rolling_max'] = df.groupby('customer_id')[feature].rolling(roll_window).max().reset_index(level=0, drop=True)
        df[f'{feature}_rolling_std'] = df.groupby('customer_id')[feature].rolling(roll_window).std().reset_index(level=0, drop=True)
    
    # 填充滚动特征的缺失值
    print("填充滚动特征缺失值...")
    rolling_features = [col for col in df.columns if 'rolling_' in col]
    df[rolling_features] = df[rolling_features].fillna(0)
    
    # 添加动态特征和计算的特征
    print("合并所有动态特征...")
    dynamic_features.extend(['asset_growth_rate', 'asset_mom_change', 'asset_cumulative_growth'] + rolling_features)
    
    print(f"最终动态特征数量：{len(dynamic_features)}")
    return df[dynamic_features]

# 构建目标变量
def build_target_variable(df):
    # 目标：未来3个月资产是否达到100万
    df = df.sort_values(['customer_id', 'stat_month'])
    
    # 计算每个客户未来3个月的最大资产
    df['future_3m_max_assets'] = df.groupby('customer_id')['total_assets'].shift(-3)
    
    # 构建二分类目标变量（1: 未来3个月资产达到100万，0: 否则）
    df['target_high_value'] = (df['future_3m_max_assets'] >= 1000000).astype(int)
    
    # 移除最后3个月的数据（无法计算未来3个月的资产）
    df = df.dropna(subset=['future_3m_max_assets'])
    
    return df[['target_high_value']]

# 特征编码和缩放
def encode_and_scale_features(X_static, X_dynamic):
    # 分离数值特征和类别特征
    numeric_features = X_static.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_static.select_dtypes(include=['object']).columns.tolist()
    
    # 数值特征处理流水线
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 类别特征处理流水线
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 合并静态特征处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 处理静态特征
    X_static_processed = preprocessor.fit_transform(X_static)
    
    # 处理动态特征（只需要标准化）
    dynamic_scaler = StandardScaler()
    X_dynamic_processed = dynamic_scaler.fit_transform(X_dynamic)
    
    # 合并静态和动态特征
    X_processed = np.hstack([X_static_processed, X_dynamic_processed])
    
    return X_processed

# 主函数
def main():
    print("开始特征工程处理...")
    
    # 预处理数据
    df = preprocess_data()
    print(f"数据预处理完成，数据形状：{df.shape}")
    
    # 查看数据中的时间范围
    df['stat_month'] = pd.to_datetime(df['stat_month'])
    print(f"数据时间范围：{df['stat_month'].min()} 至 {df['stat_month'].max()}")
    
    # 处理静态特征
    static_features = process_static_features(df)
    print(f"静态特征处理完成，特征数量：{len(static_features.columns)}")
    
    # 处理动态特征
    dynamic_features = process_dynamic_features(df)
    print(f"动态特征处理完成，特征数量：{len(dynamic_features.columns)}")
    
    # 构建目标变量
    target = build_target_variable(df)
    print(f"目标变量构建完成，目标分布：{target['target_high_value'].value_counts().to_dict()}")
    
    # 确保索引一致
    static_features = static_features.loc[target.index]
    dynamic_features = dynamic_features.loc[target.index]
    
    # 特征编码和缩放
    X_processed = encode_and_scale_features(static_features, dynamic_features)
    print(f"特征编码和缩放完成，处理后特征数量：{X_processed.shape[1]}")
    
    # 保存处理后的特征
    # 获取特征名称
    static_numeric_features = static_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    static_categorical_features = static_features.select_dtypes(include=['object']).columns.tolist()
    
    # 创建编码器获取类别特征名称
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(static_features[static_categorical_features])
    onehot_feature_names = encoder.get_feature_names_out(static_categorical_features).tolist()
    
    # 合并所有特征名称
    feature_names = static_numeric_features + onehot_feature_names + dynamic_features.columns.tolist()
    
    # 创建结果数据框
    result_df = pd.DataFrame(X_processed, columns=feature_names)
    result_df['target_high_value'] = target.values
    result_df['customer_id'] = df.loc[target.index, 'customer_id'].values
    result_df['stat_month'] = df.loc[target.index, 'stat_month'].dt.strftime('%Y-%m').values
    
    # 原始完整特征保存
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"完整特征数据保存到：{output_path}")
    
    # 按照时间顺序分割训练集和测试集
    print("\n开始按时间顺序分割训练集和测试集...")
    
    # 将stat_month转换为datetime类型用于比较
    result_df['stat_month_dt'] = pd.to_datetime(result_df['stat_month'])
    
    # 定义时间分割点
    # 假设数据包含2024-07至2024-12的数据
    # 训练集：使用2024-07至2024-08的特征，预测2024-09至2024-11的结果
    # 测试集：使用2024-09至2024-10的特征，预测2024-10至2024-12的结果
    train_start = '2024-07'
    train_end = '2024-08'
    test_start = '2024-09'
    test_end = '2024-10'
    
    # 生成训练集
    train_mask = (result_df['stat_month'] >= train_start) & (result_df['stat_month'] <= train_end)
    df_train = result_df[train_mask].copy()
    df_train = df_train.drop('stat_month_dt', axis=1)
    
    # 生成测试集
    test_mask = (result_df['stat_month'] >= test_start) & (result_df['stat_month'] <= test_end)
    df_test = result_df[test_mask].copy()
    df_test = df_test.drop('stat_month_dt', axis=1)
    
    # 保存训练集和测试集
    train_output_path = os.path.join(base_path, 'train_features.csv')
    test_output_path = os.path.join(base_path, 'test_features.csv')
    
    df_train.to_csv(train_output_path, index=False, encoding='utf-8')
    df_test.to_csv(test_output_path, index=False, encoding='utf-8')
    
    print(f"训练集保存到：{train_output_path}")
    print(f"训练集形状：{df_train.shape}")
    print(f"训练集目标分布：{df_train['target_high_value'].value_counts().to_dict()}")
    
    print(f"测试集保存到：{test_output_path}")
    print(f"测试集形状：{df_test.shape}")
    print(f"测试集目标分布：{df_test['target_high_value'].value_counts().to_dict()}")
    
    print("\n特征工程完成！")
    print("\n特征工程结果概览：")
    print(result_df.head())

if __name__ == "__main__":
    main()