import pandas as pd
import os

# 设置文件路径
base_path = os.path.dirname(__file__)
customer_base_path = os.path.join(base_path, 'customer_base.csv')
customer_behavior_path = os.path.join(base_path, 'customer_behavior_assets.csv')

# 读取customer_base.csv并显示前5行
print("=== customer_base.csv 完整字段信息 ===")
try:
    df_base = pd.read_csv(customer_base_path)
    print(f"数据形状: {df_base.shape}")
    print(f"\n列名完整列表:")
    for i, col in enumerate(df_base.columns, 1):
        print(f"{i}. {col}")
    print(f"\n数据类型:")
    print(df_base.dtypes)
    print(f"\n前5行数据:")
    print(df_base.head(5))
except Exception as e:
    print(f"读取customer_base.csv失败: {e}")

print("\n" + "="*80 + "\n")

# 读取customer_behavior_assets.csv并显示前5行
print("=== customer_behavior_assets.csv 完整字段信息 ===")
try:
    df_behavior = pd.read_csv(customer_behavior_path)
    print(f"数据形状: {df_behavior.shape}")
    print(f"\n列名完整列表:")
    for i, col in enumerate(df_behavior.columns, 1):
        print(f"{i}. {col}")
    print(f"\n数据类型:")
    print(df_behavior.dtypes)
    print(f"\n前5行数据:")
    print(df_behavior.head(5))
except Exception as e:
    print(f"读取customer_behavior_assets.csv失败: {e}")
