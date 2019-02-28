import os

import numpy as np
from myelin.metric import MetricClient
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.utils import shuffle

# from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets, svm, metrics
digits = datasets.load_digits()


data_path = os.environ.get('DATA_PATH') or '/tmp/data/'

if not os.path.exists(data_path):
    os.makedirs(data_path)

# mnist = input_data.read_data_sets("MNIST_data/")


np.save(os.path.join(data_path, "train_data.npy"), digits.images)
np.save(os.path.join(data_path, "train_labels.npy"), digits.target)


data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

mnist_images = np.load(os.path.join(data_path, "train_data.npy"))
mnist_labels = np.load(os.path.join(data_path, "train_labels.npy"))

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(mnist_images)
data = mnist_images.reshape((n_samples, -1))
targets = mnist_labels

data, targets = shuffle(data, targets)
classifier = RandomForestClassifier(n_estimators=30)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], targets[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = targets[n_samples // 2:]
test_data = data[n_samples // 2:]

print(classifier.score(test_data, expected))

predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

joblib.dump(classifier, os.path.join(model_path, 'sk.pkl'))
accuracy = metrics.accuracy_score(expected, predicted)

# self.namespace = os.environ["NAMESPACE"]
# self.task_id = os.environ["TASK_ID"]
# self.axon_name = os.environ["AXON_NAME"]
# self.url = "http://{}-prometheus-pushgateway:{}/metrics/job/{}/pod/".format(self.namespace, self.port, self.task_id)

metric_client = MetricClient()
metric_client.post_update("accuracy", accuracy)