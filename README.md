# Adversarial-Examples-on-Meshes
This repository contains the code for my Master's Thesis, which consists in developing a way to generalize adversarial perturbations on meshes.

## Dependencies
### PyTorch 1.4
This project uses **PyTorch** in order to compute gradients and use accelerated hardware, hence a version of pytorch equal to **1.4** should be used.

### Other packages
All other packages can be installed by running

	pip install -r requirements.txt

### CUDA KNN (optional)
In order to compute the nearest neighbors and chamfer distance, a CUDA implementation of KNN is used. It can be installed using 
	
	git clone https://github.com/unlimblue/KNN_CUDA.git
	cd KNN_CUDA
	make && make install
It is necessary when using pointnet, otherwise this dependency can be skipped.

## Dataset Data
The data for the dataset can be downloaded [here](http://faust.is.tue.mpg.de/) for Faust and [here](https://coma.is.tue.mpg.de/) for CoMA (only the registered scan are necessary). Once downloaded put the data inside a subfolder named `raw` inside a root folder choosen by the user, *e.g.* if  `/home/faust` is the root directory for the dataset, then the downloaded data should be put in `/home/faust/raw`).

## Pre-trained Parameters
The pre-trained parametes for the classifier can be found [here](https://drive.google.com/drive/folders/1L6lwZO4M8JXw5IOgyNMhpMj5JRWO4oEw?usp=sharing).
Pass this data in input when instantiating a classifier.
 