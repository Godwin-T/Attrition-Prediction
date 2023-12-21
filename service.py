import pandas as pd
import requests

url = "http://127.0.0.1:5080/predict"
data_path = "./bct-data-summit/test.csv"

data_frame = pd.read_csv(data_path)
data_dict = data_frame.to_dict()

response = requests.post(url, json=data_dict).json()
ids = response["id"]
prediction = [float(i) for i in response["Attrition"]]
dicts = {"id": ids, "Attrition": prediction}
output_frame = pd.DataFrame(dicts)
# output_frame.to_csv("./Attrition.csv", index=False)
print(output_frame.shape)
