import os.path
from collections import Counter

import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

import misc.faust as faust
import models

FAUST = "../../Downloads/Mesh-Datasets/MyFaustDataset"
MODEL_PATH = "../model_data/data.pt"

dataset = faust.FaustDataset(FAUST)
num_classes=10

model = models.ChebClassifier(
    param_conv_layers=[64,64,32,32],
    E_t=dataset.downscale_matrices,
    D_t=dataset.downscaled_edges,
    num_classes = num_classes)

#model = models.StupidNet(6890*3,num_classes=10, hidden_layer_features=256)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)


#dataset.shuffle() TODO look why shuffling doesn't seem to work
traindata = dataset[:80]
evaldata = dataset[0:20]

if not os.path.exists(MODEL_PATH):
    # meters
    loss_values = []

    # train module
    model.train()
    epoch_number=15
    for epoch in range(epoch_number):
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for data in tqdm.tqdm(traindata):
            optimizer.zero_grad()
            out = model(data.pos)
            out = out.view(1,-1)

            loss = criterion(out, data.y)
            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

    # save the model's parameters
    torch.save(model.state_dict(), MODEL_PATH)

    #plot loss function ----------------#
    import matplotlib.pyplot as plot    #
    fig = plot.figure()                 #
    loss_values = np.array(loss_values) #
    plot.plot(loss_values)              #
    plot.show()                         #
    #-----------------------------------#

# load the model parameters
model.load_state_dict(torch.load(MODEL_PATH))

#evaluation process

model.eval()
incorrected_classes = dict()
correct = 0
for data in tqdm.tqdm(evaldata):
    optimizer.zero_grad()
    out:torch.Tensor = model(data.pos)
    _, prediction = out.max(dim=0)
    target = int(data.y)
    if target == prediction:
        correct +=1
    else:
        if target not in incorrected_classes:
            incorrected_classes[target] = Counter()
        incorrected_classes[target][prediction] +=1

print(correct/len(evaldata))



# Try to understand underlying network
