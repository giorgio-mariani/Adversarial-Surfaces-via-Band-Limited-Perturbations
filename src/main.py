import dataset
import models
import train
'''
FAUST = "../../Downloads/Mesh-Datasets/MyFaustDataset"
COMA = "../../Downloads/Mesh-Datasets/MyComaDataset"
SHREC14 =  "../../Downloads/Mesh-Datasets/MyShrec14"

PARAMS_FILE = "../model_data/FAUST10.pt"


traindata = dataset.FaustDataset(FAUST, train=True, test=False)
traindata = dataset.FaustAugmented(FAUST, train=True, test=False)
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

import adversarial.carlini_wagner as cw
# targeted attack using C&W method
logger = cw.ValueLogger({"adversarial": lambda x:x.adversarial_loss()})
builder = cw.CWBuilder(search_iterations=1)
builder.set_classifier(model).set_mesh(x,e,f).set_target(t)

builder.set_distortion_function(cw.L2_distortion).set_perturbation_type("lowband", eigs_num=eigs_num)
builder.set_minimization_iterations(0).set_adversarial_coeff(0.1)
adex_cw = builder.build(usetqdm="standard")

# untargeted attack using FGSM
adex_it = pgd.FGSMBuilder().set_classifier(model).set_mesh(x,e,f).build()
'''

# built-in libraries
import os 

# third party libraries
import matplotlib.pyplot as plt 
import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

# repository modules
import models
import train
import adversarial.carlini_wagner as cw
import adversarial.pgd as pgd
import dataset
import utils

REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath('__file__')),"..")
FAUST = os.path.join(REPO_ROOT,"datasets/faust")
PARAMS_FILE = os.path.join(REPO_ROOT, "model_data/data.pt")

traindata = dataset.FaustDataset(FAUST, train=True, test=False, transform_data=True)
testdata = dataset.FaustDataset(FAUST, train=False, test=True,  transform_data=True)

model = models.ChebnetClassifier(
    param_conv_layers=[128,128,64,64],
    D_t = traindata.downscale_matrices,
    E_t = traindata.downscaled_edges,
    num_classes = traindata.num_classes,
    parameters_file=PARAMS_FILE)

#train network
train.train(
    train_data=traindata,
    classifier=model,
    parameters_file=PARAMS_FILE,
    epoch_number=0)

'''
i =  5#random.randint(0, len(traindata)-1)
x = traindata[i].pos
e = traindata[i].edge_index.t() # needs to be transposed
f = traindata[i].face.t() # needs to be transposed
y = traindata[i].y
N=20

projections = {"lowband":pgd.lowband_filter,
              "lowband-clip":lambda a,x: pgd.clip(a, pgd.lowband_filter(a,x)),
              "lowband-clipnorm":lambda a,x: pgd.clip_norms(a,pgd.lowband_filter(a,x)),
              "none": lambda a,x :x,
              "clip":pgd.clip,
              "clipnorm":pgd.clip_norms}

if model(x).argmax() == y:
    builder = pgd.LowbandPGDBuilder().set_iterations(5).set_epsilon(0.03).set_alpha(1).set_eigs_number(36)
    builder.set_projection(projections["none"])
    builder.set_mesh(x,e,f).set_classifier(model)
    adex = builder.build(usetqdm="standard")
    print("successful: {}".format(adex.is_successful))
    print("adversarial example's prediction: {}".format(model(adex.perturbed_pos).argmax()))
    print("ground-truth: {}".format(model(adex.pos).argmax()))
else:
    print("Oh no!")'''
