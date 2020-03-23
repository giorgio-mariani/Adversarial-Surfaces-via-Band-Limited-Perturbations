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

#dataset_data = dataset.CoMADataset(COMA)


traindata = dataset.FaustDataset(FAUST, train=True, test=False)
testdata = dataset.FaustDataset(FAUST, train=False, test=True)
num_classes = traindata.num_classes


model = models.ChebnetClassifier(
    param_conv_layers=[128,128,64,64],
    D_t=traindata.downscale_matrices,
    E_t=traindata.downscaled_edges,
    num_classes = num_classes)

#train network
train.train(
    train_data=traindata,
    classifier=model,
    param_file=PARAMS_FILE,
    epoch_number=0)


#compute accuracy
accuracy, confusion_matrix = train.evaluate(eval_data=testdata,classifier=model)
print(accuracy)
import matplotlib.pyplot as plt 
plt.matshow(confusion_matrix)
#plt.show()

i=20
x = traindata[i].pos
e = traindata[i].edge_index.t()
f = traindata[i].face.t()
y = traindata[i].y
t = 2
n = x.shape[0]
eigs_num = 100

builder = cw.AdversarialExampleBuilder(model).set_log_interval(2)
builder.set_perturbation_type("spectral").set_mesh(x,e,f).set_target(t).set_distortion_functions(cw.L2_distortion)
adex = builder.set_adversarial_coeff(0.1).build(100, 8e-4, usetqdm="standard")