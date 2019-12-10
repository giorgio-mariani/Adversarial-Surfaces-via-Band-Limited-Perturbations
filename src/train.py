import os
from collections import Counter

import tqdm
import torch
from torch_geometric.data import Dataset

def train(
    train_data:Dataset, 
    classifier:torch.nn.Module,
    param_file:str,
    device:torch.device=None,
    epoch_number:int = 1):

    if os.path.exists(param_file):
        classifier.load_state_dict(torch.load(param_file))

    device = torch.device('cpu') if device is None else device

    # meters
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=5e-4)
    
    traindata_pos = [mesh.pos.to(device) for mesh in train_data]
    traindata_gtruth = [mesh.y.to(device) for mesh in train_data]

    # train module
    classifier.train()
    for epoch in range(epoch_number):
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
    device:torch.device=None):

    device = torch.device('cpu') if device is None else device

    classifier.eval()
    incorrect_classes = dict()
    correct = 0
    for mesh in tqdm.tqdm(eval_data):
        out:torch.Tensor = classifier(mesh.pos)
        _, prediction = out.max(dim=0)
        target = int(mesh.y)
        if target == prediction:
            correct +=1
        else:
            if target not in incorrect_classes:
                incorrect_classes[target] = Counter()
            incorrect_classes[target][prediction] +=1

    return correct/len(eval_data), incorrect_classes
