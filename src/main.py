import numpy as np
import tqdm
import torch 
import torch.nn.functional as func

import misc.faust as faust
import models

FAUST = "../../Downloads/Mesh-Datasets/MyFaustDataset"

paramnumber = lambda x: sum([np.prod(p.size()) for p in x])

dataset = faust.FaustDataset(FAUST)
D = dataset.downscale_matrices
E = dataset.downscaled_edges
num_classes = 10
lastlayernodes = D[-1].shape[0] #NOTE this is the number of nodes after the last convolutional layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.ChebClassifier(
    param_conv_layers=[64,64,32,32],
    dense_input_nodes = lastlayernodes,
    num_classes = num_classes).to(device)

p1 = paramnumber(model.parameters())
model = models.StupidNet(6890*3,num_classes=10, hidden_layer_features=256) #TODO remove
p2 = paramnumber(model.parameters())
print(p1)
print(p2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

# meters
loss_values = []

# train module
model.train()
epoch_number=15
#dataset.shuffle() TODO look why shuffling doesn't seem to work
traindata = dataset[:80]
evaldata = dataset[20:]
for epoch in range(epoch_number):
    print("epoch "+str(epoch+1)+" of "+str(epoch_number))
    for data in tqdm.tqdm(traindata):
        optimizer.zero_grad()
        out = model(data, E, D)
        out = out.view(1,-1)

        loss = criterion(out, data.y)
        loss_values.append(loss.item())

        loss.backward()
        optimizer.step()

#plot loss function
import matplotlib.pyplot as plot
fig = plot.figure()
loss_values = np.array(loss_values)
plot.plot(loss_values)
plot.show()

#evaluation process (TODO right now it uses evaluation set same as )
model.eval()
correct, incorrect = 0, 0
for data in tqdm.tqdm(evaldata):
    optimizer.zero_grad()
    out:torch.Tensor = model(data, E, D)
    _, argmax = out.max(dim=0)
    if data.y == argmax:
        correct +=1
    else:
        incorrect +=1

print(correct/(correct+incorrect))


# Try to understand underlying network
