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

#traindata = dataset.CoMADataset(COMA)
#traindata = dataset.FaustAugmented(FAUST, train=True, test=False)

traindata = dataset.FaustDataset(FAUST, train=True, test=False)
testdata = dataset.FaustDataset(FAUST, train=False, test=True)
num_classes = traindata.num_classes
traindata[0]

model = models.ChebnetClassifier(
    param_conv_layers=[128,128,64,64],
    D_t=traindata.downscale_matrices,
    E_t=traindata.downscaled_edges,
    num_classes = num_classes,
    parameters_file=PARAMS_FILE)

#train network
train.train(
    train_data=traindata,
    classifier=model,
    parameters_file=PARAMS_FILE,
    epoch_number=0)


#compute accuracy
accuracy, confusion_matrix = train.evaluate(eval_data=testdata,classifier=model)
print(accuracy)
import matplotlib.pyplot as plt 
plt.matshow(confusion_matrix)
#plt.show()

'''
i=20
x = traindata[i].pos
e = traindata[i].edge_index.t()
f = traindata[i].face.t()
y = traindata[i].y
t = 2
n = x.shape[0]
eigs_num = 100

builder = cw.AdversarialExampleBuilder().set_classifier(model).set_log_interval(2)
builder.set_perturbation_type("spectral").set_mesh(x,e,f).set_target(t).set_distortion_function(cw.L2_distortion)
adex = builder.set_adversarial_coeff(0.1).build(100, 8e-4, usetqdm="standard")
'''

import adversarial.uap as uap


builder = cw.AdversarialExampleBuilder().set_classifier(model).set_log_interval(2)
builder.set_perturbation_type("spectral").set_distortion_function(cw.L2_distortion)

uap.UAP_computation(
    data=testdata,
    adv_builder=builder,
    classifier=model,
    delta=0.5,
    eps=1,
    starting_coeff=1e-3,
    learning_rate=8e-4)
