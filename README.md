# CNN_GAN_Ge
Pulse shape simulation of a Ge detector using CNN GAN.

## 1. Data
Initial dataset consists of a set of charge pulses, each containing 5000 points. 

## 2. Model
The minimax GAN (with PCA preprocessing) or Wasserstein-GAN (with adversarial autoencoder preproccessing) are used to generate data.

## 3. Files
* **requirements.txt**: The list required dependencies to run the project.

* **src**: Contains scripts to tun the project.
	* **src/config**: Contains project's parameters.
	* **src/processing**: Contains class Processing which is used to preprocess data and to return processed data back to the initial state. 
	* **src/utils**: Contains functions to calculate pulses' parameters and Intersection Over Union (IoU).
	* **src/models**: Contains used models' topologies and training/inference/visualization steps.

* **data**: Contains data files.
	* **data/raw**: Contains raw data to train the GAN.
	* **data/processed**: Contains generated data after the GAN has been trained. 

* **artefacts**: Contains training artefacts (scaler/pca).

* **models**: Contains trained models.

* **notebooks**: Contains each step of the GAN project development.

* **reports/figures**: Contains figures from the training/inference steps.

* **pipeline.py**: The main file to run the project.


## 4. Usage
Dependencies installation:

```SH
pip install -r requirements.txt
```

Conda:
```SH
conda install pip
pip install -r requirements.txt
```

Parameters can be set manually in the src/config/config.py

How to run:

```SH
python3 pipeline.py 
```
