from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
from src.utils.utils import energy_calculation, current_calculation, drifttime_calculation, \
tail_slope_calculation, calculate_iou

class Dataset:
	def __init__(self, data, batch_size):
		self.data = data
		self.batch_size = batch_size

	def to_loader(self):
		"""
		Makes DataLoader
		Returns: DataLoader
		"""
		train_data_length = self.data.shape[0]
		train_data = torch.zeros((train_data_length, self.data.shape[1]))
		train_data = torch.tensor(self.data, dtype=torch.float)
		train_labels = torch.zeros(train_data_length, 1)
		train_set = [
		    (train_data[i], train_labels[i]) for i in range(train_data_length)
		]
		train_loader = torch.utils.data.DataLoader(
			train_set, batch_size=self.batch_size, shuffle=True, drop_last=True
			)
		return train_loader

class Run:
	def __init__(self, data, netG, netD, params):
		self.data = data #preprocessed
		self.netD = netD
		self.netG = netG
		self.params = params

	def data_to_loader(self):
		dataset = Dataset(self.data, batch_size=self.params.batch_size)
		return dataset.to_loader()

	def calculate_metrics(self, real, generated):
		"""
		Calculates pulses' parameters and IoU
		Input:
			real - initial dataset
			generated - postprocessed generated samples
		"""
		en_real, en_gen = energy_calculation(real), energy_calculation(generated)
		curr_real, curr_gen = current_calculation(real), current_calculation(generated)
		tail_real, tail_gen = tail_slope_calculation(real), tail_slope_calculation(generated)
		drift_real, drift_gen = drifttime_calculation(real), drifttime_calculation(generated)
		return {
		'energy' : (en_real, en_gen, round(calculate_iou(en_real, en_gen), 2)),
		'current' : (curr_real, curr_gen, round(calculate_iou(curr_real, curr_gen), 2)),
		'tail_slope' : (tail_real, tail_gen, round(calculate_iou(tail_real, tail_gen), 2)),
		'drifttime' : (drift_real, drift_gen, round(calculate_iou(drift_real, drift_gen), 2)),
		}

	def plot_hists(self, real, generated):
		"""
		Plots and saves histograms with pulses' parameters and IoU
		"""
		metrics = self.calculate_metrics(real, generated)

		fig = plt.figure()
		_, _, _ = plt.hist(metrics.get('energy')[0], bins = 20,  label = 'real', density = True, histtype = 'step', color = 'red')
		_, _, _ = plt.hist(metrics.get('energy')[1], bins = 20,  label = 'gen', density = True, histtype = 'step', color = 'black')
		plt.legend()
		plt.title('Charge amplitude: IoU = ' + str(metrics.get('energy')[2]))
		plt.xlabel('a.u.')
		fig.savefig(self.params.images_path / Path('energy_pca.png'))

		fig = plt.figure()
		_, _, _ = plt.hist(metrics.get('current')[0], bins = 20,  label = 'real', density = True, histtype = 'step', color = 'red')
		_, _, _ = plt.hist(metrics.get('current')[1], bins = 20,  label = 'gen', density = True, histtype = 'step', color = 'black')
		plt.legend()
		plt.title('Current amplitude: IoU = ' + str(metrics.get('current')[2]))
		plt.xlabel('a.u./time')
		fig.savefig(self.params.images_path / Path('current_pca.png'))

		fig = plt.figure()
		_, _, _ = plt.hist(metrics.get('tail_slope')[0], range=[0, 20], bins = 50,  label = 'real', density = True, histtype = 'step', color = 'red')
		_, _, _ = plt.hist(metrics.get('tail_slope')[1], range=[0, 20], bins = 50,  label = 'gen', density = True, histtype = 'step', color = 'black')
		plt.legend()
		plt.title('Tail_slope: IoU = ' + str(metrics.get('tail_slope')[2]))
		plt.xlabel('a.u./time')
		fig.savefig(self.params.images_path / Path('tail_slope_pca.png'))

		fig = plt.figure()
		_, _, _ = plt.hist(metrics.get('drifttime')[0], range=[0, 100], bins = 50,  label = 'real', density = True, histtype = 'step', color = 'red')
		_, _, _ = plt.hist(metrics.get('drifttime')[1], range=[0, 100], bins = 50,  label = 'gen', density = True, histtype = 'step', color = 'black')
		plt.legend()
		plt.title('Drift time: IoU = ' + str(metrics.get('drifttime')[2]))
		plt.xlabel('time')
		fig.savefig(self.params.images_path / Path('drifttime_pca.png'))

	@staticmethod
	def plot_losses(loss_A, loss_A_name, loss_B, loss_B_name, title, xlabel, ylabel, savepath):
		"""
		Plots loss functions
		"""
		fig = plt.figure()
		plt.plot(loss_A, '.', label=loss_A_name)
		plt.plot(loss_B, '.',  label=loss_B_name)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.legend('upper right')
		fig.savefig(savepath / Path(title + '.png'))

	def pca_train(self):
		"""
		Training for PCA-GAN
		"""
		criterion = nn.BCELoss()
		optimizerD = torch.optim.SGD(self.netD.parameters(), lr=self.params.lr)
		optimizerG = torch.optim.SGD(self.netG.parameters(), lr=self.params.lr)
		G_losses = []
		D_losses = []
		loader = self.data_to_loader()
		for epoch in range(self.params.epochs):
		    for i, (x, y) in enumerate(loader):
		        real_samples_labels = torch.ones((self.params.batch_size, 1))
		        latent_space_samples = torch.randn((self.params.batch_size, self.params.random_dim))
		        generated_samples = self.netG(latent_space_samples)
		        generated_samples_labels = torch.zeros((self.params.batch_size, 1))
		        all_samples = torch.cat((x, generated_samples))
		        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
		        
		        ## training the discriminator
		        self.netD.zero_grad()
		        output_discriminator = self.netD(all_samples)
		        
		        errD = criterion(output_discriminator, all_samples_labels)
		        errD.backward()
		        optimizerD.step()
		        
		        ## training the generator
		        latent_space_samples = torch.randn(self.params.batch_size, self.params.random_dim)
		        self.netG.zero_grad()
		        generated_samples = self.netG(latent_space_samples)
		        outputDG = self.netD(generated_samples)
		        errG = criterion(outputDG, real_samples_labels)
		        errG.backward()
		        optimizerG.step()
		        
		        G_losses.append(errG.item()) 
		        D_losses.append(errD.item())
		        if (epoch % 5 == 0 and i + 1 == len(loader)):
		        	print(f"Epoch{epoch} Loss D.: {errD}" "||" f"Loss G.: {errG}")

		pickle.dump(self.netD, open(self.params.models_path / 'pca_netD.pkl', 'wb'))
		pickle.dump(self.netG, open(self.params.models_path / 'pca_netG.pkl', 'wb'))
		self.plot_losses(G_losses, 'G',  D_losses, 'D', 'losses_PCA', 'iteration', 'loss', self.params.images_path)

	def pca_generate(self):
		"""
		Inference for PCA-GAN
		"""
		netG = pickle.load(open(self.params.models_path / 'pca_netG.pkl', 'rb'))
		random_samples = torch.randn(self.params.num_samples, self.params.random_dim)
		return netG(random_samples).detach().numpy()
