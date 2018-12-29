import os

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'

mnist = input_data.read_data_sets("MNIST_data/")

np.save(os.path.join(data_path, "train_data.npy"), mnist.train.images)
np.save(os.path.join(data_path, "train_labels.npy"), mnist.train.labels)
