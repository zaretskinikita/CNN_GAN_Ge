from dataclasses import dataclass
from pathlib import Path

@dataclass
class Parameters:
   # General parameters
   project_path: Path = Path('/Users/nzaretski/Desktop/gerda/CNN_GAN_Ge')
   input_data_path: Path = project_path / 'data/raw/data.pkl' # path to raw data
   output_data_path: Path = project_path / 'data/processed/' # path to generated data
   artefacts_path: Path = project_path / 'artefacts' # path where artefacts are saved
   images_path: Path = project_path / 'reports/figures' # path where figures are saved
   models_path: Path = project_path / 'models' # path where trained models are saved
   num_samples: int = 1000 # number of generated samples
   random_seed: int = 99

   model_type: str = 'wgan' # or 'wgan'
   mode: str = 'train' # or generate

   # PCA-GAN parameters 
   if model_type == 'pca':
      ## Model parameters
      seq_len: int = 94 # number of components for PCA
      random_dim: int = 23 # random dimension, from which the Generator model starts
         
      ## Training parameters
      epochs: int = 119
      batch_size: int = 128
      lr: float = 0.03453234251828275

   # WGAN parameters 
   elif model_type == 'wgan':
      ## Model parameters
      init_len = 5000 # initial space
      seq_len = 321 # encoded space
      random_dim = 22  # latent space for the generator model
      prior_dim = 64 # for the prior discriminator model (autoencoder)
         
      ## Training parameters
      batch_size = 128
      lr_wgan = 0.0028886418155533253 # learning rate
      lr_auto = 0.006435632456005054
      epochs_autoen = 87 #epochs to train autoencoder
      ITERS = 1031 # iterations to train WGAN
      CRITIC_ITERS = 8
      LAMBDA = 6 # penalty
