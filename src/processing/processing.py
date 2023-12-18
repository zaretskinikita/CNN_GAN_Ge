import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class Processing:
	def __init__(self, params):
		self.params = params
		self.input_data_path = params.input_data_path
		self.output_data_path = params.output_data_path
		self.artefacts_path = params.artefacts_path
		self.data = pickle.load(open(self.input_data_path, 'rb'))
		self.scaler = None
		self.pca = None

	def pca_preprocessing(self):
		"""
		PCA and StScaler transformation of the initial data
		Returns: transformed data + saves artifacts
		"""
		self.scaler = StandardScaler()
		self.pca = PCA(n_components=self.params.seq_len)
		pickle.dump({'pca': self.pca, 'scaler': self.scaler}, 
			open(self.artefacts_path / 'gan_pca.pkl', 'wb'))
		return self.pca.fit_transform(self.scaler.fit_transform(self.data))

	def pca_postprocessing(self, data):
		"""
		PCA and StScaler inverse transformation
		Returns: inverse-transformed data + saves inverse-transformed data
		"""
		data = self.scaler.inverse_transform(self.pca.inverse_transform(data))
		pickle.dump(data, open(self.output_data_path, 'wb'))
		return data
