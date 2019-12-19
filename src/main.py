import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

import faust 
import models
import train
import adversarial.adversarial_generator as adv
import mesh.laplacian

FAUST = "../../Downloads/Mesh-Datasets/MyFaustDataset"
PARAMS_FILE = "../model_data/data.pt"

dataset = faust.FaustDataset(FAUST)
num_classes=10

model = models.ChebClassifier(
    param_conv_layers=[64,64,32,32],
    D_t=dataset.downscale_matrices,
    E_t=dataset.downscaled_edges,
    num_classes = num_classes)
    
traindata = dataset[:80]
evaldata = dataset[80:]
'''
#train network
train.train(
    train_data=traindata,
    classifier=model,
    param_file=PARAMS_FILE,
    epoch_number=0)


#compute accuracy
accuracy, confusion_matrix = train.evaluate(
    eval_data=evaldata,
    classifier=model)
'''

pos = dataset[0].pos
faces = dataset[0].face

r = torch.zeros(size=[pos.shape[0], 3])#torch.normal(0, std=0.001, size=[pos.shape[0], 3], requires_grad=True, device=pos.device)
x = (pos+r).to(torch.double)
mesh.laplacian.LB_v2(x, faces.t())
