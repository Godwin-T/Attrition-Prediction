import requests
import pandas as pd

url = "http://127.0.0.1:5080/predict"
data_path = "../../data/bct-data-summit/test.csv"
response = requests.post(url, json=data_path).json()
print(response)
