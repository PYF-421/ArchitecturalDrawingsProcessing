import requests
import json

url = "http://192.168.224.62:8080/cad/json/"

jwt_token = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJhZG1pbiIsImxvZ2luX3VzZXJfa2V5IjoiMTdmZDcyNzYtY2EyYy00MjljLWEwYzItNDUxYjhhNmU3MzlhIn0.CjyBx77tDFkCl5qFuwo76nOiCmnX0oW_Z2anstyAAu-9TWlT5hf_zU0DSMLDXWCsR1WZYHY_ijPip5ar7cet4w"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Authorization": f"Bearer {jwt_token}"  # 最推荐的方式
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        print("请求成功！")
        try:
            data = response.json()
            print(json.dumps(data, indent=4, ensure_ascii=False))
        except ValueError:
            print("返回内容不是JSON：")
            print(response.text)
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print("响应内容：")
        print(response.text)
        
except Exception as e:
    print(f"错误：{e}")