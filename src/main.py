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

i=20
x = dataset_data[i].pos
e = dataset_data[i].edge_index.t()
f = dataset_data[i].face.t()
y = dataset_data[i].y
n = x.shape[0]
eigs_num = 300

import scipy
(si, sv), (ai, av) = mesh.laplacian.LB_v2(pos=x, faces=f)
ri, ci = si.cpu().detach().numpy()
sv = sv.cpu().detach().numpy()
S = scipy.sparse.csr_matrix( (sv, (ri,ci)), shape=(n,n))

ri,ci = ai.cpu().detach().numpy()
av = av.cpu().detach().numpy()
A = scipy.sparse.csr_matrix( (av, (ri,ci)), shape=(n,n))
e, phi = scipy.sparse.linalg.eigsh(S, M=A, k=eigs_num, sigma=-1e-6)

eigvals = torch.tensor(e, device=x.device, dtype=x.dtype)
eigvecs = torch.tensor(phi, device=x.device, dtype=x.dtype)

print(eigvals.shape)
print(eigvecs.shape)