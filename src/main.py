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
import random

import adversarial.pgd as pgd
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


from os import mkdir,listdir
from os.path import join, split

dirs = [join(REPO_ROOT,"pgd_tests_l2_lowband"),
join(REPO_ROOT,"pgd_tests_l2"),
join(REPO_ROOT,"pgd_tests_sign_lowband"),
join(REPO_ROOT,"pgd_tests_sign"),
join(REPO_ROOT,"pgd_tests_l2_lowband-clipnorm"),
join(REPO_ROOT,"pgd_tests_l2_clipnorm"),
join(REPO_ROOT,"pgd_tests_sign_lowband-clipnorm"),
join(REPO_ROOT,"pgd_tests_sign_clipnorm")]

import tqdm
import torch_sparse as tsparse
import matplotlib.pyplot as plt


def load_adex(filename):
    obj = np.load(filename, allow_pickle=True)
    obj = np.reshape(obj, [1])[0]
    return obj

def meancurvature_distance(directory):
    c, s, l2 = [],[], []
    for path in tqdm.tqdm(listdir(directory)):
        if path.split(".")[-1] == "npy":
            obj = load_adex(join(directory,path))
            ppos = torch.tensor(obj["perturbed-positions"])
            pos = torch.tensor(obj["positions"])
            n = ppos.shape[0]
            faces = torch.tensor(obj["faces"])
            stiff, mass = utils.laplacebeltrami_FEM_v2(pos, faces)
            stiff_p, mass_p = utils.laplacebeltrami_FEM_v2(ppos, faces)

            tmp = tsparse.spmm(*stiff, n, n, pos)
            perturbed_tmp = tsparse.spmm(*stiff_p, n, n, ppos)
                
            ai, av = mass
            ai_r, av_r = mass_p

            mcf = tsparse.spmm(ai, torch.reciprocal(av), n, n, tmp)
            perturbed_mcf = tsparse.spmm(ai_r, torch.reciprocal(av_r), n, n, perturbed_tmp)
            diff_curvature = mcf.norm(dim=-1,p=2) - perturbed_mcf.norm(dim=-1, p=2)
            curvature_dist = (av*diff_curvature**2).sum().sqrt().item()
            c.append(curvature_dist)
            s.append(obj["success"])
            l2.append(obj["l2"])
    return c, s, l2

for directory in dirs:
    c1, s1, l21 = meancurvature_distance(directory)

    print(directory+": ")
    print("mean-curvature difference: ", sum(c1)/len(c1))
    print("success-rate:",sum(s1)/len(s1))
    print("l2-distance:", sum(l21)/len(l21))
    print()
