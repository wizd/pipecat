import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

def delete_old_rooms():
    # 加载环境变量
    load_dotenv()
    api_key = os.getenv('DAILY_API_KEY')
    
    if not api_key:
        print("错误：未找到DAILY_API_KEY环境变量")
        return
    
    # API配置
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    base_url = 'https://api.daily.co/v1'
    
    try:
        # 获取所有rooms
        response = requests.get(f'{base_url}/rooms', headers=headers)
        response.raise_for_status()
        rooms = response.json()
        
        # 计算一天前的时间
        one_day_ago = datetime.utcnow() - timedelta(days=1)
        
        # 遍历所有rooms
        for room in rooms.get('data', []):
            created_at = datetime.strptime(room['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
            room_name = room['name']
            
            # 如果room创建时间超过一天
            if created_at < one_day_ago:
                print(f"删除过期room: {room_name} (创建于 {created_at})")
                # 删除room
                delete_response = requests.delete(
                    f'{base_url}/rooms/{room_name}',
                    headers=headers
                )
                if delete_response.status_code == 200:
                    print(f"成功删除room: {room_name}")
                else:
                    print(f"删除room失败: {room_name}, 状态码: {delete_response.status_code}")
            else:
                print(f"保留room: {room_name} (创建于 {created_at})")
                
    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {str(e)}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == '__main__':
    delete_old_rooms()
