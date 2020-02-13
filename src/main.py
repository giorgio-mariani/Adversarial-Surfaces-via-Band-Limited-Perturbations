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

coma_data = dataset.ComaDataset(COMA)
num_classes=coma_data.num_classes

model = models.ChebClassifier(
    param_conv_layers=[64,64,32,32],
    D_t=coma_data.downscale_matrices,
    E_t=coma_data.downscaled_edges,
    num_classes = num_classes)
    
sep = int(0.8*len(coma_data))
traindata = coma_data[:sep]
evaldata = coma_data[sep:]

#train network
train.train(
    train_data=traindata,
    classifier=model,
    param_file=PARAMS_FILE,
    epoch_number=3)


#compute accuracy
accuracy, confusion_matrix = train.evaluate(
    eval_data=evaldata,
    classifier=model)

pos = coma_data[0].pos
faces = coma_data[0].face

r = torch.zeros(size=[pos.shape[0], 3])#torch.normal(0, std=0.001, size=[pos.shape[0], 3], requires_grad=True, device=pos.device)
x = (pos+r).to(torch.double)