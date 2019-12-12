import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

import misc.faust as faust
import models
import train
import adversarial.adversarial_generator as adv

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

#train network
train.train(
    train_data=traindata,
    classifier=model,
    param_file=PARAMS_FILE,
    epoch_number=1)

#compute accuracy
accuracy, confusion_matrix = train.evaluate(
    eval_data=evaldata,
    classifier=model)

# choose target class for adversarial perturbations
print(accuracy)
maxval, maxkey = 0,0
for key in confusion_matrix.keys():
    tmp = sum(confusion_matrix[key].values())
    if tmp>=maxval:
        maxval = tmp
        maxkey = key
target = maxkey

print("choosing class "+str(target)+" for targeted adversarial examples.")

mesh = dataset[23]
c, r = adv.find_perturbation(
    x=mesh.pos,
    edge_index=mesh.edge_index,
    target=target,
    classifier=model,
    bsearch_iterations=10,
    optim_iterations=100)

r = r.detach().numpy()
np.save("perturbation", [r,c])
