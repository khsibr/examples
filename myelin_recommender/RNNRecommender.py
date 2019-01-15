from sklearn.externals import joblib
import os
from myelin_recommender.cf_model import CFModel
from myelin_recommender.utils import load_obj
import numpy as np

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class RNNRecommender(object):
	def __init__(self):
		model_parameters = load_obj(model_path, "model_parameters")
		self.trained_model = CFModel(model_parameters["max_userid"], model_parameters["max_movieid"],
									 model_parameters["k_factors"])
		self.trained_model.load_weights(os.path.join(model_path, 'weights.h5'))
		self.class_names = ["class:rating"]

	def predict_rating(self, row):
		user_id = row[0]
		movie_id = row[1]
		return self.trained_model.rate(user_id - 1, movie_id - 1)

	def predict(self, X, feature_names):
		predictions = np.apply_along_axis(self.predict_rating, axis=1, arr=X)
		return predictions