import os
from collections import Counter

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import tqdm

import adversarial.pgd as pgd

def train(
    train_data:Dataset, 
    classifier:torch.nn.Module,
    parameters_file:str,
    epoch_number:int = 1,
    learning_rate:float=1e-3):
        
    # meters
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=5e-4)

    # train module
    classifier.train()
    for epoch in range(epoch_number):
        # start epoch
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for i in tqdm.trange(len(train_data)):
            x = train_data[i].pos
            y = train_data[i].y

            optimizer.zero_grad()
            out = classifier(x)
            out = out.view(1,-1)

            loss = criterion(out, y)
            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

    # save the model's parameters
    torch.save(classifier.state_dict(), parameters_file)
    return loss_values

def evaluate(
    eval_data:Dataset, 
    classifier:torch.nn.Module,
    epoch_number=1):

    classifier.eval()
    evaldata_pos = [mesh.pos for mesh in eval_data]
    evaldata_gtruth = [mesh.y.item() for mesh in eval_data]

    confusion = None
    for epoch in range(epoch_number):
        for i in tqdm.trange(len(eval_data)):
            x = evaldata_pos[i]
            y = evaldata_gtruth[i]

            out:torch.Tensor = classifier(x)
            if confusion is None:
                num_classes = out.shape[-1]
                confusion = torch.zeros([num_classes, num_classes])
            
            _, prediction = out.max(dim=-1)
            target = int(y)
            confusion[target, prediction] +=1
            
            correct = torch.diag(confusion).sum()
            accuracy = correct/confusion.sum()
    return accuracy, confusion


#------------------------------------------------------------------------------

def PGD_train(
    train_data:Dataset, 
    classifier:torch.nn.Module,
    parameters_file:str,
    epoch_number:int = 1,
    learning_rate:float=1e-3,
    steps=10,
    eps=0.001,
    alpha=0.045):
        
    # meters
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=5e-4)

    # train module
    classifier.train()
    for epoch in range(epoch_number):
        # start epoch
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for i in tqdm.trange(len(train_data)):
            # create adversarial example
            builder = pgd.PGDBuilder().set_classifier(classifier)
            device = train_data[i].pos.device
            builder.set_mesh(
                train_data[i].pos,
                train_data[i].edge_index.t().to(device), 
                train_data[i].face.t().to(device))
            builder.set_iterations(steps).set_epsilon(eps).set_alpha(alpha).set_eigs_number(36)
            x = builder.build().perturbed_pos
            y = train_data[i].y

            optimizer.zero_grad()
            out = classifier(x)
            out = out.view(1,-1)

            loss = criterion(out, y)
            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

    # save the model's parameters
    torch.save(classifier.state_dict(), parameters_file)
    return loss_values

#------------------------------------------------------------
def train_SHREC14(
    train_data:Dataset,
    classifier:torch.nn.Module,
    parameters_file:str,
    epoch_number:int = 1,
    learning_rate:float=1e-3):
        
    # meters
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=5e-4)

    # train module
    classifier.train()
    for epoch in range(epoch_number):
        # start epoch
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for i in tqdm.trange(len(train_data)):
            device = train_data[i].pos.device
            optimizer.zero_grad()
            out = classifier(
                train_data[i], 
                [(i.to(device),v.to(device), s) for (i,v,s) in train_data.downscale_matrices[i]],
                [ e .to(device) for e in train_data.downscaled_edges[i]]
                )
            out = out.view(1,-1)

            loss = criterion(out, train_data[i].y)
            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

    # save the model's parameters
    torch.save(classifier.state_dict(), parameters_file)
    return loss_values


def evaluate_SHREC14(
    eval_data:Dataset, 
    classifier:torch.nn.Module,
    epoch_number=1):

    classifier.eval()
    evaldata_gtruth = [mesh.y.item() for mesh in eval_data]

    confusion = None
    for epoch in range(epoch_number):
        for i in tqdm.trange(len(eval_data)):
            y = evaldata_gtruth[i]
            device =eval_data[i].pos.device
            out = classifier(
                eval_data[i],
                [(i.to(device),v.to(device), s) for (i,v,s) in eval_data.downscale_matrices[i]],
                [e .to(device) for e in eval_data.downscaled_edges[i]])

            if confusion is None:
                num_classes = out.shape[-1]
                confusion = torch.zeros([num_classes, num_classes])
            
            _, prediction = out.max(dim=-1)
            target = int(y)
            confusion[target, prediction] +=1
            
    correct = torch.diag(confusion).sum()
    accuracy = correct/confusion.sum()
    return accuracy, confusion

def PGD_train_SHREC14(
    train_data:Dataset,
    classifier:torch.nn.Module,
    parameters_file:str,
    epoch_number:int = 1,
    learning_rate:float=1e-3,
    steps=10,
    alpha=0.001,
    eps=0.01):
        
    # meters
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=5e-4)

    def get_classifier(data, mesh_index, device, model): #IMPORTANT THIS IS A TRICK, NOT A GOOD PRACTICE
        I = [(i.to(device),v.to(device), s) for (i,v,s) in data.downscale_matrices[mesh_index]]
        E = [ e .to(device) for e in data.downscaled_edges[mesh_index]]
        return  lambda x:  model(Data(pos=x, edge_index=data[mesh_index].edge_index, y=data[mesh_index].y), I, E)

    # train module
    classifier.train()
    for epoch in range(epoch_number):
        # start epoch
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for i in tqdm.trange(len(train_data)):
            device = train_data[i].pos.device
            
            builder = pgd.PGDBuilder().set_classifier(get_classifier(train_data, i, device, classifier))
            builder.set_mesh(
                train_data[i].pos,
                train_data[i].edge_index.t().to(device), 
                train_data[i].face.t().to(device))
            builder.set_iterations(steps).set_epsilon(eps).set_alpha().set_eigs_number(36)
            builder.set_projection(pgd.lowband_filter)
            x = builder.build().perturbed_pos

            mesh = Data(pos = x, edge_index=train_data[i].edge_index.t(), y=train_data[i].y)
            optimizer.zero_grad()
            out = classifier(
                mesh, 
                [(i.to(device),v.to(device), s) for (i,v,s) in train_data.downscale_matrices[i]],
                [ e .to(device) for e in train_data.downscaled_edges[i]])
            out = out.view(1,-1)

            loss = criterion(out, train_data[i].y)
            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

    # save the model's parameters
    torch.save(classifier.state_dict(), parameters_file)
    return loss_values