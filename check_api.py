import requests
import json

# 测试资产分析API
def check_asset_analysis_api():
    url = 'http://127.0.0.1:5000/api/asset_analysis'
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    
    # 获取原始响应文本
    raw_text = response.text
    print(f"\nRaw Response Text:")
    print(raw_text)
    
    # 尝试解析JSON
    try:
        data = response.json()
        print(f"\nParsed JSON Data:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        
        # 检查资产构成数据
        print(f"\nAsset Composition Data:")
        print(json.dumps(data['asset_composition'], ensure_ascii=False, indent=2))
        
        # 检查资产等级分布数据
        print(f"\nAsset Level Distribution Data:")
        print(json.dumps(data['asset_level_dist'], ensure_ascii=False, indent=2))
        
    except json.JSONDecodeError as e:
        print(f"\nJSON Decode Error: {e}")
        print(f"Error position: {e.pos}")
        print(f"Error line: {raw_text.splitlines()[e.lineno-1]}")

if __name__ == '__main__':
    check_asset_analysis_api()