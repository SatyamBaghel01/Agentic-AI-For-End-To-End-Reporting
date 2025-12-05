import requests

url = "http://127.0.0.1:8000/query"

payload = {
    "user_question": "What was the average spindle speed yesterday?",
    "username": "guest"
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())
