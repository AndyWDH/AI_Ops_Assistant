import pandas as pd
import os
import json

# 设置文件路径
base_path = os.path.dirname(__file__)
customer_base_path = os.path.join(base_path, 'customer_base.csv')
customer_behavior_path = os.path.join(base_path, 'customer_behavior_assets.csv')

# 读取数据
df_base = pd.read_csv(customer_base_path)
df_behavior = pd.read_csv(customer_behavior_path)

# 测试资产等级分布数据
def test_asset_level_data():
    print("Testing asset_level distribution data...")
    asset_level_dist = df_behavior['asset_level'].value_counts().reset_index()
    asset_level_dist.columns = ['asset_level', 'count']
    
    print("Raw data:")
    print(asset_level_dist)
    print("\nData types:")
    print(asset_level_dist.dtypes)
    
    # 转换为字典
    asset_level_dict = asset_level_dist.to_dict('records')
    print("\nDictionary format:")
    print(asset_level_dict)
    
    # 转换为JSON
    asset_level_json = json.dumps(asset_level_dict, ensure_ascii=False, indent=2)
    print("\nJSON format:")
    print(asset_level_json)
    
    # 测试完整的asset_analysis数据结构
    print("\nFull asset_analysis data structure:")
    full_data = {
        'asset_level_dist': asset_level_dict,
        'asset_composition': [],
        'asset_group_dist': []
    }
    full_json = json.dumps(full_data, ensure_ascii=False, indent=2)
    print(full_json)

# 测试资产构成数据
def test_asset_composition_data():
    print("\n" + "="*50 + "\n")
    print("Testing asset composition data...")
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
    
    print("Raw data:")
    print(asset_composition)
    print("\nData types:")
    print(asset_composition.dtypes)
    
    # 转换为字典
    asset_composition_dict = asset_composition.to_dict('records')
    print("\nDictionary format:")
    print(asset_composition_dict)
    
    # 转换为JSON
    asset_composition_json = json.dumps(asset_composition_dict, ensure_ascii=False, indent=2)
    print("\nJSON format:")
    print(asset_composition_json)

if __name__ == '__main__':
    test_asset_level_data()
    test_asset_composition_data()