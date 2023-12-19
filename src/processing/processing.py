import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch

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
		data_preprocessed = self.pca.fit_transform(self.scaler.fit_transform(self.data))
		pickle.dump({'pca': self.pca, 'scaler': self.scaler}, 
			open(self.artefacts_path / 'gan_pca.pkl', 'wb'))
		return data_preprocessed

	def pca_postprocessing(self, data):
		"""
		PCA and StScaler inverse transformation
		Input: generated data
		Returns: inverse-transformed data + saves inverse-transformed data
		"""
		artefacts = pickle.load(open(self.artefacts_path / 'gan_pca.pkl', 'rb'))
		scaler = artefacts.get('scaler')
		pca = artefacts.get('pca')
		data_post = scaler.inverse_transform(pca.inverse_transform(data))
		pickle.dump(data_post, open(self.output_data_path / 'pca_gan_data.pkl', 'wb'))
		return data_post

	def wgan_preprocessing(self):
		"""
		StScaler transformation of the initial data
		Returns: transformed data + saves artifacts
		"""
		self.scaler = StandardScaler()
		data_preprocessed = self.scaler.fit_transform(self.data)
		pickle.dump({'scaler': self.scaler}, 
			open(self.artefacts_path / 'wgan.pkl', 'wb'))
		return data_preprocessed

	def wgan_postprocessing(self, data):
		"""
		Transformation of the wgan-generated output: 
			decoding generated data with the pretrained decoder model + 
			StScaler inverse transformation
		Input: generated data
		Returns: decoded and inverse-transformed data + saves transformed data
		"""
		decoder = pickle.load(open(self.params.models_path / 'wgan_decoder.pkl', 'rb'))
		artefacts = pickle.load(open(self.artefacts_path / 'wgan.pkl', 'rb'))
		scaler = artefacts.get('scaler')
		data_post = scaler.inverse_transform(decoder(data).detach().numpy())
		pickle.dump(data_post, open(self.output_data_path / 'wgan_data.pkl', 'wb'))
		return data_post
