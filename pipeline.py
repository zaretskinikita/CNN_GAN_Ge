from src.processing.processing import Processing
from src.config.config import Parameters
from src.models.models import Generator, Discriminator, Decoder, Encoder, PriorDiscriminator
from src.models.run import Run
import torch
import random
torch.manual_seed(Parameters.random_seed)
random.seed(Parameters.random_seed)


class Pipeline(Parameters):
	def __init__(self):
		if Parameters.mode == 'train':
			if Parameters.model_type == 'pca':
				processing = Processing(Parameters)
				data_processed = processing.pca_preprocessing()
				netD = Discriminator(Parameters)
				netG = Generator(Parameters)

				# training
				run = Run(
					data=data_processed, 
					netG=netG, 
					netD=netD, 
					encoder=None, 
					decoder=None,
					disc=None, 
					params=Parameters
					)
				run.pca_train()

				# generated data in latent space
				generated = run.pca_generate()
				# generated data in initial space
				generated = processing.pca_postprocessing(generated)
				# histograms with initial data and generated samples in initial space
				run.plot_hists(processing.data, generated, 'pca')
				run.plot_pulses(pulses=processing.data, model_type='pca', title='Real', color='red')
				run.plot_pulses(pulses=generated, model_type='pca', title='Generated', color='black')

			elif  Parameters.model_type == 'wgan':
				processing = Processing(Parameters)
				data_processed = processing.wgan_preprocessing()
				netD = Discriminator(Parameters)
				netG = Generator(Parameters)
				decoder = Decoder(Parameters)
				encoder = Encoder(Parameters)
				disc = PriorDiscriminator(Parameters)

				# training
				run = Run(
					data=data_processed, 
					netG=netG, 
					netD=netD, 
					encoder=encoder, 
					decoder=decoder,
					disc=disc, 
					params=Parameters
					)
				run.autoencoder_train()
				run.wgan_train()

				# generated data in latent space
				generated = run.wgan_generate()
				# generated data in initial space
				generated = processing.wgan_postprocessing(generated)
				# histograms with initial data and generated samples in initial space
				run.plot_hists(processing.data, generated, 'wgan')
				run.plot_pulses(pulses=processing.data, model_type='wgan', title='Real', color='red')
				run.plot_pulses(pulses=generated, model_type='wgan', title='Generated', color='black')

		elif Parameters.mode == 'generate':
			if Parameters.model_type == 'pca':
				processing = Processing(Parameters)
				run = Run(
					data=None, 
					netG=None, 
					netD=None, 
					encoder=None, 
					decoder=None,
					disc=None, 
					params=Parameters
					)
				generated = run.pca_generate()
				generated = processing.pca_postprocessing(generated)
				run.plot_hists(processing.data, generated, 'pca')
				run.plot_pulses(pulses=processing.data, model_type='pca', title='Real', color='red')
				run.plot_pulses(pulses=generated, model_type='pca', title='Generated', color='black')

			elif  Parameters.model_type == 'wgan':
				processing = Processing(Parameters)
				run = Run(
					data=None, 
					netG=None, 
					netD=None, 
					encoder=None, 
					decoder=None,
					disc=None, 
					params=Parameters
					)
				generated = run.wgan_generate()
				generated = processing.wgan_postprocessing(generated)
				run.plot_hists(processing.data, generated, 'wgan')
				run.plot_pulses(pulses=processing.data, model_type='wgan', title='Real', color='red')
				run.plot_pulses(pulses=generated, model_type='wgan', title='Generated', color='black')


if __name__ == '__main__':
	pipeline = Pipeline()
