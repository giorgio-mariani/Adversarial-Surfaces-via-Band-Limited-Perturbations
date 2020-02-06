import os
from collections import Counter

import tqdm
import torch
from torch_geometric.data import Dataset

import mesh.transforms

def train(
    train_data:Dataset, 
    classifier:torch.nn.Module,
    param_file:str,
    device:torch.device=None,
    epoch_number:int = 1,
    rotate=False):

    if os.path.exists(param_file):
        classifier.load_state_dict(torch.load(param_file))

    device = torch.device('cpu') if device is None else device

    # meters
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=5e-4)
    
    traindata_gtruth = [mesh.y.to(device) for mesh in train_data]
    traindata_pos = [mesh.pos.to(device) for mesh in train_data]

    # train module
    classifier.train()
    for epoch in range(epoch_number):
        # randomize position and rotation of mesh
        for x in traindata_pos :
            if translate: mesh.transforms.transform_translation_(x)
            if rotate: mesh.transforms.transform_rotation_(x)
        
        # start epoch
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for i in tqdm.trange(len(train_data)):
            x = traindata_pos[i]
            y = traindata_gtruth[i]

            optimizer.zero_grad()
            out = classifier(x)
            out = out.view(1,-1)

            loss = criterion(out, y)
            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

    # save the model's parameters
    torch.save(classifier.state_dict(), param_file)
    return loss_values

def evaluate(
    eval_data:Dataset, 
    classifier:torch.nn.Module,
    device:torch.device=None,
    rotate=True,
    epoch_number=1):

    device = torch.device('cpu') if device is None else device

    classifier.eval()
    incorrect_classes = dict()
    correct = 0

    torch.zeros([])

    evaldata_pos = [mesh.pos.to(device) for mesh in eval_data]
    evaldata_gtruth = [mesh.y.item() for mesh in eval_data]

    confusion = None
    for epoch in range(epoch_number):
        for x in evaldata_pos :
            if translate: mesh.transforms.transform_translation_(x)
            if rotate: mesh.transforms.transform_rotation_(x)

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
    return accuracy, incorrect_classes
