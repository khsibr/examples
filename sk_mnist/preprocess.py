import os

import numpy as np
from sklearn import datasets

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'

if not os.path.exists(data_path):
    os.makedirs(data_path)

digits = datasets.load_digits()

np.save(os.path.join(data_path, "train_data.npy"), digits.images)
np.save(os.path.join(data_path, "train_labels.npy"), digits.target)
