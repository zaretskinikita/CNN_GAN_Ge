from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
from src.utils.utils import energy_calculation, current_calculation, drifttime_calculation, \
tail_slope_calculation, calculate_iou, inf_train_gen, calc_gradient_penalty

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
	def __init__(self, data, netG, netD, encoder, decoder, disc, params):
		self.data = data #preprocessed
		self.netD = netD
		self.netG = netG
		self.encoder = encoder
		self.decoder = decoder
		self.disc = disc
		self.params = params

	def data_to_loader(self):
		dataset = Dataset(self.data, batch_size=self.params.batch_size)
		return dataset.to_loader()

	def calculate_metrics(self, real, generated):
		"""
		Calculates pulses' parameters and IoU
		Input:
			real: initial dataset
			generated: postprocessed generated samples
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

	def plot_hists(self, real, generated, model_type):
		"""
		Plots and saves histograms with pulses' parameters and IoU
		Input:
			real: initial dataset
			generated: postprocessed generated samples
		"""
		metrics = self.calculate_metrics(real, generated)

		fig = plt.figure()
		_, _, _ = plt.hist(metrics.get('energy')[0], bins = 20,  label = 'real', density = True, histtype = 'step', color = 'red')
		_, _, _ = plt.hist(metrics.get('energy')[1], bins = 20,  label = 'gen', density = True, histtype = 'step', color = 'black')
		plt.legend()
		plt.title('Charge amplitude: IoU = ' + str(metrics.get('energy')[2]))
		plt.xlabel('a.u.')
		fig.savefig(self.params.images_path / Path('energy_' + model_type + '.png'))

		fig = plt.figure()
		_, _, _ = plt.hist(metrics.get('current')[0], bins = 20,  label = 'real', density = True, histtype = 'step', color = 'red')
		_, _, _ = plt.hist(metrics.get('current')[1], bins = 20,  label = 'gen', density = True, histtype = 'step', color = 'black')
		plt.legend()
		plt.title('Current amplitude: IoU = ' + str(metrics.get('current')[2]))
		plt.xlabel('a.u./time')
		fig.savefig(self.params.images_path / Path('current_' + model_type + '.png'))

		fig = plt.figure()
		_, _, _ = plt.hist(metrics.get('tail_slope')[0], range=[0, 20], bins = 50,  label = 'real', density = True, histtype = 'step', color = 'red')
		_, _, _ = plt.hist(metrics.get('tail_slope')[1], range=[0, 20], bins = 50,  label = 'gen', density = True, histtype = 'step', color = 'black')
		plt.legend()
		plt.title('Tail_slope: IoU = ' + str(metrics.get('tail_slope')[2]))
		plt.xlabel('a.u./time')
		fig.savefig(self.params.images_path / Path('tail_slope_' + model_type + '.png'))

		fig = plt.figure()
		_, _, _ = plt.hist(metrics.get('drifttime')[0], range=[0, 100], bins = 50,  label = 'real', density = True, histtype = 'step', color = 'red')
		_, _, _ = plt.hist(metrics.get('drifttime')[1], range=[0, 100], bins = 50,  label = 'gen', density = True, histtype = 'step', color = 'black')
		plt.legend()
		plt.title('Drift time: IoU = ' + str(metrics.get('drifttime')[2]))
		plt.xlabel('time')
		fig.savefig(self.params.images_path / Path('drifttime_' + model_type + '.png'))

	@staticmethod
	def plot_losses(loss_A, loss_A_name, loss_B, loss_B_name, title, xlabel, ylabel, savepath):
		"""
		Plots loss functions
		"""
		fig = plt.figure()
		plt.plot(loss_A, '.', label=loss_A_name)
		if loss_B is not None:
			plt.plot(loss_B, '.',  label=loss_B_name)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.legend('upper right')
		fig.savefig(savepath / Path(title + '.png'))

	def pca_train(self):
		"""
		Training for PCA-GAN
		Uses preprocessed data
		"""
		print('---PCA---')
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
		Returns: not postprocessed output of the generator model
		"""
		netG = pickle.load(open(self.params.models_path / 'pca_netG.pkl', 'rb'))
		random_samples = torch.randn(self.params.num_samples, self.params.random_dim)
		return netG(random_samples).detach().numpy()

	def autoencoder_train(self):
		"""
		Training for wgan
		Uses preprocessed data
		"""
		print('---AUTOENCODER---')
		train_loader = self.data_to_loader()
		recon_loss = nn.MSELoss(reduction = 'sum')
		optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.params.lr_auto)
		optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.params.lr_auto)

		optim_encoder_reg = torch.optim.Adam(self.encoder.parameters(), lr=self.params.lr_auto)
		optim_D = torch.optim.Adam(self.disc.parameters(), lr=self.params.lr_auto)

		losses = []
		one = torch.FloatTensor([1])
		mone = one * -1
		for epoch in range(self.params.epochs_autoen):
		    for i, (x, y) in enumerate(train_loader):
		        batch = x.size(0)
		        self.encoder.train()
		        self.decoder.train()
		        self.disc.train()
		        self.encoder.zero_grad()
		        self.decoder.zero_grad()
		        self.disc.zero_grad()

		        # Reconstruction phase
		        z = self.encoder(x)
		        x_hat = self.decoder(z)
		        loss = recon_loss(x_hat,x)
		        loss.backward()
		        optim_encoder.step()
		        optim_decoder.step()

		        # Discriminator phase
		        self.encoder.zero_grad()
		        self.decoder.zero_grad()
		        self.disc.zero_grad()
		        self.encoder.eval()
		        z_real_gauss = autograd.Variable(torch.randn(z.size())*1)
		        z_fake_gauss = self.encoder(x)
		        D_real_gauss, D_fake_gauss = self.disc(z_real_gauss), self.disc(z_fake_gauss.detach())
		        real_outputs = D_real_gauss.mean(dim=0)
		        real_outputs.backward(mone)
		        fake_outputs = D_fake_gauss.mean(dim=0)
		        fake_outputs.backward(one)
		        optim_D.step()
		        
		        # Regularization phase
		        self.encoder.zero_grad()
		        self.decoder.zero_grad()
		        self.disc.zero_grad()
		        self.encoder.train()
		        z = self.encoder(x)
		        D_fake_gauss = self.disc(z)
		        fake_outputs = D_fake_gauss.mean(dim=0)
		        fake_outputs.backward(mone)
		        optim_encoder_reg.step()
		        if i % 10 == 0:
		            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		                epoch, i * len(x), len(train_loader.dataset),
		                100. * i / len(train_loader), loss.item()))
		        losses.append(loss.item())
		
		pickle.dump(self.decoder, open(self.params.models_path / 'wgan_decoder.pkl', 'wb'))
		pickle.dump(self.encoder, open(self.params.models_path / 'wgan_encoder.pkl', 'wb'))
		self.plot_losses(losses, 'G', None, 'D', 'losses_autoencoder', 'iteration', 'loss', self.params.images_path)

	def wgan_train(self):
		"""
		Training for wgan
		Uses preprocessed data and encodes it with the pretrained encoder
		"""
		print('---WGAN---')
		optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.params.lr_wgan)
		optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.params.lr_wgan)
		loss_G = []
		loss_D = []
		one = torch.tensor(1, dtype=torch.float)
		mone = one * -1
		train_loader = self.data_to_loader()
		data = inf_train_gen(train_loader)
		for iteration in range(self.params.ITERS):
		    # (1) Update D network
		    for p in self.netD.parameters():  # reset requires_grad
		        p.requires_grad = True  # they are set to False below in netG update

		    for iter_d in range(self.params.CRITIC_ITERS):
		        _data = next(data)
		        _data = _data
		        # encoding
		        _data = self.encoder(_data).detach()
		        real_data = torch.Tensor(_data)
		        real_data = real_data
		        real_data_v = autograd.Variable(real_data)
		        self.netD.zero_grad()
		        # train with real
		        D_real = self.netD(real_data_v)
		        D_real = D_real.mean()
		        D_real.backward(mone)
		        # train with fake
		        noise = torch.randn(self.params.batch_size, self.params.random_dim)
		        fake = self.netG(noise).detach()
		        inputv = fake
		        D_fake = self.netD(inputv)
		        D_fake = D_fake.mean()
		        D_fake.backward(one)
		        # train with gradient penalty
		        gradient_penalty = calc_gradient_penalty(self.netD, real_data_v.data, fake.data, 
		        	self.params.batch_size, self.params.LAMBDA)
		        gradient_penalty.backward()
		        D_cost = D_fake - D_real + gradient_penalty
		        Wasserstein_D = D_real - D_fake
		        optimizerD.step()
		    loss_D.append(D_cost.item())

		    # (2) Update G network
		    for p in self.netD.parameters():
		        p.requires_grad = False  # to avoid computation
		    self.netG.zero_grad()
		    _data = next(data)
		    noise = torch.randn(self.params.batch_size, self.params.random_dim)
		    noisev = autograd.Variable(noise)
		    fake = self.netG(noisev)
		    G = self.netD(fake)
		    G = G.mean()
		    G.backward(mone)
		    G_cost = -G
		    loss_G.append(G_cost.item())
		    optimizerG.step()
		    if iteration % 50 == 0:
		        print(G_cost.item(), D_cost.item())

		pickle.dump(self.netD, open(self.params.models_path / 'wgan_netD.pkl', 'wb'))
		pickle.dump(self.netG, open(self.params.models_path / 'wgan_netG.pkl', 'wb'))
		self.plot_losses(loss_G, 'G',  loss_D, 'D', 'losses_wgan', 'iteration', 'loss', self.params.images_path)

	def wgan_generate(self):
		"""
		Inference for WGAN
		Returns: not postprocessed output of the generator model
		"""
		netG = pickle.load(open(self.params.models_path / 'wgan_netG.pkl', 'rb'))
		random_samples = torch.randn(self.params.num_samples, self.params.random_dim)
		return netG(random_samples).detach()
