import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

import adversarial.carlini_wagner as cw
import adversarial.pgd as pgd
import dataset
import models
import train

SHREC14 =  "../../Downloads/Mesh-Datasets/MyShrec14"
PARAMS_FILE = "../model_data/SHREC14.pt"

traindata = dataset.Shrec14Dataset(SHREC14, train=True, test=False)
testdata = dataset.Shrec14Dataset(SHREC14, train=False, test=True)

#data.transform = lambda x:x

num_dense_units = 16*traindata.downscale_matrices[0][-1][-1][0]
model = models.chebynet.ChebnetClassifier_SHREC14(
        nums_conv_units=[32,32,16,16],
        num_classes=traindata.num_classes,
        num_dense_units=num_dense_units,
        parameters_file=PARAMS_FILE)

#train network
train.train_SHREC14(
    train_data=traindata,
    classifier=model,
    parameters_file=PARAMS_FILE,
    epoch_number=1,
    learning_rate=3e-4)


#evaluate network
accuracy, confusion_matrix = train.evaluate_SHREC14(
    eval_data=testdata,
    classifier=model,
    epoch_number=1)

print(accuracy)
import matplotlib.pyplot as plt
plt.matshow(confusion_matrix)
plt.show()



FAUST = "../../Downloads/Mesh-Datasets/MyFaustDataset"
COMA = "../../Downloads/Mesh-Datasets/MyComaDataset"
SHREC14 =  "../../Downloads/Mesh-Datasets/MyShrec14"

PARAMS_FILE = "../model_data/FAUST10.pt"


traindata = dataset.FaustDataset(FAUST, train=True, test=False)
testdata = dataset.FaustDataset(FAUST, train=False, test=True)

model = models.ChebnetClassifier(
    param_conv_layers=[128,128,64,64],
    D_t=traindata.downscale_matrices,
    E_t=traindata.downscaled_edges,
    num_classes = traindata.num_classes,
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

i = 20
x = traindata[i].pos
e = traindata[i].edge_index.t()
f = traindata[i].face.t()
y = traindata[i].y
t = 2
n = x.shape[0]
eigs_num = 100

# targeted attack using C&W method
logger = cw.ValueLogger({"adversarial": lambda x:x.adversarial_loss()})
builder = cw.CWBuilder()
builder.set_classifier(model).set_mesh(x,e,f).set_target(t)

builder.set_distortion_function(cw.LB_distortion).set_perturbation_type("spectral", eigs_num=eigs_num)
builder.set_minimization_iterations(0).set_adversarial_coeff(0.1)
adex_cw = builder.build(usetqdm="standard")

# untargeted attack using FGSM
adex_it = pgd.FGSMBuilder().set_classifier(model).set_mesh(x,e,f).build()

