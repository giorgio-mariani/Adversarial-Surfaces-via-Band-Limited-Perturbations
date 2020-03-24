# Adversarial-Examples-on-Meshes
This repository contains the code for my Master's Thesis, which consists in developing a way to generalize adversarial perturbations on meshes.

## Dependencies
This project uses **PyTorch** in order to compute gradients and use accelerated hardware, hence a version of pytorch greater than or equal to 1.4.0 should be used.
All other packages can be installed by running

	pip install -r requirements.txt

## Dataset Data
The data for the dataset can be downloaded [here](http://faust.is.tue.mpg.de/) for Faust and [here](https://coma.is.tue.mpg.de/) for CoMA (only the registered scan are necessary). Once downloaded put the data inside a subfolder named `raw` inside a root folder choosen by the user, *e.g.* if  `/home/faust` is the root directory for the dataset, then the downloaded data should be put in `/home/faust/raw`).

## Pre-trained Parameters
The pre-trained parametes for the classifier can be found [here](https://drive.google.com/drive/folders/1L6lwZO4M8JXw5IOgyNMhpMj5JRWO4oEw?usp=sharing).
Pass this data in input when instantiating a classifier.
 