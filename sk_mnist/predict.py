import json
import os

import numpy as np
import requests
import imageio

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
mnist_images = np.load(os.path.join(data_path, "train_data.npy"))

url = 'http://localhost:5000/predict'
image = mnist_images[20, :]

jsondata = json.dumps({"data": {"ndarray": [image.tolist()]}})
payload = {'json': jsondata}
imageio.imwrite('outfile.jpg', image.reshape([28, 28]))

session = requests.session()
response = session.post(url, data=payload, headers={'User-Agent': 'test'})
json_data = json.loads(response.text)
prediction = json_data["data"]["ndarray"]

print("predicted class: %s" % np.argmax(np.array(prediction)))
