import requests
import json

# 测试客户分布API
def test_customer_distribution():
    url = 'http://127.0.0.1:5000/api/customer_distribution'
    response = requests.get(url)
    print(f"Customer Distribution API Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Gender Distribution Data:")
        print(json.dumps(data['gender_dist'], indent=2, ensure_ascii=False))
        print("\nAge Distribution Data:")
        print(json.dumps(data['age_dist'], indent=2, ensure_ascii=False))
        print("\nCity Distribution Data:")
        print(json.dumps(data['city_dist'], indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")

# 测试资产分析API
def test_asset_analysis():
    url = 'http://127.0.0.1:5000/api/asset_analysis'
    response = requests.get(url)
    print(f"Asset Analysis API Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Asset Composition Data:")
        print(json.dumps(data['asset_composition'], indent=2, ensure_ascii=False))
        print("\nAsset Level Distribution Data:")
        print(json.dumps(data['asset_level_dist'], indent=2, ensure_ascii=False))
        print("\nAsset Group Distribution Data:")
        print(json.dumps(data['asset_group_dist'], indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")

if __name__ == '__main__':
    test_customer_distribution()
    test_asset_analysis()