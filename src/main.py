import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

import models
import train
import adversarial.carlini_wagner as cw
import mesh.laplacian
import dataset
import utils

FAUST = "../../Downloads/Mesh-Datasets/MyFaustDataset"
COMA = "../../Downloads/Mesh-Datasets/MyComaDataset"
PARAMS_FILE = "../model_data/dataCOMA.pt"
PARAMS_FILE = "../model_data/data.pt"

#coma_data = dataset.ComaDataset(COMA)
dataset_data = dataset.FaustDataset(FAUST)
num_classes=dataset_data.num_classes

model = models.ChebnetClassifier(
    param_conv_layers=[128,128,64,64],
    D_t=dataset_data.downscale_matrices,
    E_t=dataset_data.downscaled_edges,
    num_classes = num_classes)
    
traindata = dataset_data[20:]
evaldata = dataset_data[:20]

#train network
train.train(
    train_data=traindata,
    classifier=model,
    param_file=PARAMS_FILE,
    epoch_number=0)


#compute accuracy
accuracy, confusion_matrix = train.evaluate(eval_data=evaldata,classifier=model)
print(accuracy)
import matplotlib.pyplot as plt 
plt.matshow(confusion_matrix)
#plt.show()

i=20
x = dataset_data[i].pos
e = dataset_data[i].edge_index.t()
f = dataset_data[i].face.t()
y = dataset_data[i].y
t = 2
n = x.shape[0]
eigs_num = 100

builder = cw.AdversarialExampleBuilder(model).set_log_interval(2)
builder.set_perturbation_type("spectral").set_mesh(x,e,f).set_target(t).set_distortion_functions(cw.L2_distortion)
adex = builder.set_adversarial_coeff(0.1).build(100, 8e-4, usetqdm="standard")
