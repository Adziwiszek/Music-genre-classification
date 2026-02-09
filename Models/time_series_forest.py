from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import numpy as np

class EnsembleModel(BaseEstimator):
	def __init__(self, model, ts_model, w=0.5):
		self.model = model
		self.pipeline = Pipeline([
			("scaler", StandardScaler()),
			('clf', self.model),
		]) 
		self.ts_model = ts_model
		self.w = w

	def fit(self, X, y):
		ts_columns = [col for col in X.columns if col.startswith('ts_')]
		ts_X = X[ts_columns]
		X = X.drop(columns=ts_columns)
		self.pipeline.fit(X, y)
		self.ts_model.fit(ts_X, y)

		self.classes_ = np.unique(y)
		return self

	def predict_proba(self, X):
		ts_columns = [col for col in X.columns if col.startswith('ts_')]
		ts_X = X[ts_columns]
		X = X.drop(columns=ts_columns)
		proba1 = self.pipeline.predict_proba(X)
		proba2 = self.ts_model.predict_proba(ts_X)
		return self.w * proba1 + (1 - self.w) * proba2
	
	def predict(self, X):
		proba = self.predict_proba(X)
		return np.argmax(proba, axis=1)
	
