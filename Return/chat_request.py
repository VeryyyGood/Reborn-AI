import requests

url = "http://127.0.0.1:5000/api/chat"
data = {"message": "안녕? 넌 누구니?"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers, stream=True)

buffer = bytes()  
for byte in response.iter_content(chunk_size=1):
    if byte:
        buffer += byte 
        try:
            text = buffer.decode('utf-8') 
            print(text, end='', flush=True)  
            buffer = bytes()  
        except UnicodeDecodeError:
            continue  
