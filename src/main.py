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
import torch

# local modules
import dataset
import models
import train

SHREC14 =  "../../Downloads/Mesh-Datasets/MyShrec14"
PARAMS_FILE = "../model_data/SHREC14.pt"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
traindata = dataset.Shrec14Dataset(SHREC14, device=DEVICE, train=True, test=False)
testdata = dataset.Shrec14Dataset(SHREC14, device=DEVICE, train=False, test=True)

MODEL = models.chebynet.ChebnetClassifier_SHREC14(
        nums_conv_units=[32,32,16,16],
        num_classes=traindata.num_classes,
        parameters_file=PARAMS_FILE)
MODEL.to(DEVICE)


#train network
train.train_SHREC14(
    train_data=traindata,
    classifier=MODEL,
    parameters_file=PARAMS_FILE,
    learning_rate=3e-4,
    epoch_number=0)

import adversarial.carlini_wagner as cw
from torch_geometric.data import Data

def get_classifier(data, mesh_index): #IMPORTANT THIS IS A TRICK, NOT A GOOD PRACTICE
    I = [(i.to(DEVICE),v.to(DEVICE), s) for (i,v,s) in data.downscale_matrices[mesh_index]]
    E = [ e .to(DEVICE) for e in data.downscaled_edges[mesh_index]]
    return  lambda x:  MODEL(Data(pos=x, edge_index=data[mesh_index].edge_index, y=data[mesh_index].y), I, E)

def CW_adversarial_example(
    mesh_index=0,
    target_class=None,
    data=testdata,
    perturbation="lowband",
    distortion="local_euclidean",
    eigenvecs_number=36,
    adversarial_coefficient:float="default",
    regularization_coefficient:float="default",
    learning_rate:float=5e5,
    minimization_iterations=1000,
    tuning_iterations=3):
    """
    Create an adversarial example using the C&W method.
    
    Arguments:
     - data: the dataset containing the mesh used during the adversarial attack.
     - mesh_index: index (in data) of the mesh to perturb.
     - target_class: class goal for the targeted adversarial attack.
     - perturbation: type of perturbation, assumes values 'lowband' or 'vertex'
     - distortion: type of distortion, assumes values 'L2' or 'local_euclidean'.
    """
    
    #check input consistency:
    if perturbation not in  ["lowband","vertex"]:
        raise ValueError("Invalid input for argument 'perturbation'. Must either be 'lowband' or 'vertex'!")
        
    if distortion not in  ["L2","local_euclidean"]:
        raise ValueError("Invalid input for argument 'distortion'. Must either be 'L2' or 'local_euclidean'!")
        
    if adversarial_coefficient != "default" and not isinstance(adversarial_coefficient, float):
        raise ValueError("Invalid input for argument 'adversarial_coefficient'. Must either be the string 'default' or any floating point number!")
    
    if regularization_coefficient != "default" and not isinstance(regularization_coefficient, float):
        raise ValueError("Invalid input for argument 'regularization_coefficient'. Must either be the string 'default' or any floating point number!")

    if adversarial_coefficient == "default":
        if distortion == "L2":
            adversarial_coefficient = 5e-3
        elif distortion == "local_euclidean":
            adversarial_coefficient = 5e-7
    
    if regularization_coefficient == "default":
        if distortion == "L2":
            regularization_coefficient = 0
        elif distortion == "local_euclidean":
            regularization_coefficient = 1e3
    
    i = mesh_index
    x = data[i].pos
    e = data[i].edge_index.t().to(DEVICE) # needs to be transposed
    f = data[i].face.t().to(DEVICE) # needs to be transposed
    y = data[i].y
    t = target_class

    # configure adversarial example components
    builder = cw.CWBuilder(search_iterations=tuning_iterations)
    builder.set_classifier(get_classifier(data, i))
    builder.set_perturbation_type(perturbation, eigs_num=eigenvecs_number)
    print(adversarial_coefficient)
    builder.set_mesh(x, e, f).set_adversarial_coeff(adversarial_coefficient)
    if t is not None: builder.set_target(t)
    
    if distortion=="L2":
        builder.set_distortion_function(cw.L2_distortion)
    elif distortion=="local_euclidean":
        builder.set_distortion_function(cw.LocallyEuclideanDistortion(K=40))
        builder.set_regularization_function(cw.centroid_regularizer)

    builder.set_minimization_iterations(minimization_iterations).set_learning_rate(learning_rate)
    adex = builder.build(usetqdm="standard")
    print("adversarial attack: "+("successful" if adex.is_successful else "unsuccessful"))

CW_adversarial_example(
    data=traindata,
    mesh_index=0,
    target_class=1,
    perturbation="lowband",
    learning_rate=1e4, 
    minimization_iterations=100)