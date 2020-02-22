import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

import models
import train
import adversarial.adversarial_generator as adv
import mesh.laplacian
import dataset

FAUST = "../../Downloads/Mesh-Datasets/MyFaustDataset"
COMA = "../../Downloads/Mesh-Datasets/MyComaDataset"
PARAMS_FILE = "../model_data/dataCOMA.pt"
PARAMS_FILE = "../model_data/data.pt"

#coma_data = dataset.ComaDataset(COMA)
dataset_data = dataset.FaustDataset(FAUST)
num_classes=dataset_data.num_classes

model = models.ChebClassifier(
    param_conv_layers=[64,64,32,32],
    D_t=dataset_data.downscale_matrices,
    E_t=dataset_data.downscaled_edges,
    num_classes = num_classes)
    
sep = int(0.8*len(dataset_data))
traindata = dataset_data[:sep]
evaldata = dataset_data[sep:]

#train network
train.train(
    train_data=traindata,
    classifier=model,
    param_file=PARAMS_FILE,
    epoch_number=0)


#compute accuracy
#accuracy, confusion_matrix = train.evaluate(eval_data=evaldata,classifier=model)
#print(accuracy)

import adversarial.generators as adv
i=20
x = dataset_data[i].pos
e = dataset_data[i].edge_index.t()
f = dataset_data[i].face.t()
y = dataset_data[i].y
t = (y+1)%num_classes

gen = adv.AdversarialGenerator(
    pos=x,
    edges=e,
    faces=f,
    target=t,
    classifier=model,
    adversarial_coeff=100)

r, loss = gen.generate(iter_num=10, track_metrics=True)