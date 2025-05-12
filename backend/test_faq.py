import requests

url = "http://localhost:5000/api/faq"
payload = {
    "question": "hello",
    "lang": "en"
}

response = requests.post(url, json=payload)
print("Response:", response.json())
