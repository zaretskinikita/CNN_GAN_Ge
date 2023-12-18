from src.processing.processing import Processing
from src.config.config import Parameters
from src.models.models import Generator, Discriminator
from src.models.run import Run
import torch
import random
torch.manual_seed(Parameters.random_seed)
random.seed(Parameters.random_seed)


class Pipeline(Parameters):
	def __init__(self):
		if Parameters.model_type == 'pca':
			processing = Processing(Parameters)
			data_processed = processing.pca_preprocessing()
			netD = Discriminator(Parameters)
			netG = Generator(Parameters)

			# training
			run = Run(data_processed, netG, netD, Parameters)
			run.pca_train()

			# generated data in latent space
			generated = run.pca_generate()
			# generated data in initial space
			generated = processing.pca_postprocessing(generated)

			# histograms with initial data and generated samples in initial space
			run.plot_hists(processing.data, generated)


if __name__ == '__main__':
	pipeline = Pipeline()
