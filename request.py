import requests

url = 'http://localhost:9000/predict_api'
r = requests.post(url,json={'R&D Spend':165349, 'Administration':136897, 'Marketing Spend':471784, 'State': 1})

print(r.json())