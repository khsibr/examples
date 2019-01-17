import json
import os

import numpy as np
import requests

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
mnist_images = np.load(os.path.join(data_path, "train_data.npy"))

address = "35.234.144.51:80"
endpoint = "ml-recommender-all24-2642664409-2539877511"

url = "http://%s/%s/predict" % (address, endpoint)
data = [[10, 2], [10, 3]]

jsondata = json.dumps({"data": {"ndarray": data}})
payload = {'json': jsondata}

response = requests.post(url, data=payload, headers={'User-Agent': 'test'})
print(response.status_code)
print(response.text)
# json_data = json.loads(response.text)
# prediction = json_data["data"]["ndarray"]
# print(prediction)