import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import sys

file_specifier = 'v15'
model_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/{}/model_{}_epoch_{}.pt'
tain_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/train_{}.pt'.format(file_specifier)
val_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/val_{}.pt'.format(file_specifier)
loss_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/loss_{}.h5'.format(file_specifier)

with h5py.File('/cds/home/i/isele/tmox51020/scratch/recon/trn_data_v4.h5', 'r') as hf:
    x = np.array(hf.get('Msol1s_diff')[:-1000])
    y = np.array(hf.get('Ets_interp_ap')[:-1000])
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Device: {}'.format(device))

# move data from numpy to torch
x_tensor = torch.from_numpy(x).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)
# create tensor dataset from tensor
dataset = TensorDataset(x_tensor, y_tensor)

# split data into train and eval
train_dataset, val_dataset = random_split(dataset, [90000, 9000])
# create data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=50)
val_loader = DataLoader(dataset=val_dataset, batch_size=50)

image_size = 64

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # Code adjusted based on model
            nn.Linear(image_size*image_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.to(device)

alpha = 1e-1
batch_size = 50

loss_fun = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

def train(model, loss_fun, optimizer, x_batch, y_batch):
    model.train()
    y_pred = model(x_batch)
    loss = loss_fun(y_batch, y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

# train model
# modified from https://subscription.packtpub.com/book/data/9781800209848/11/ch11lvl1sec81/mini-batch-sgd-with-pytorch

n_epochs = 1000

losses = []
v_losses = []
for epoch in range(n_epochs):
    print(epoch)
    losses_temp = []
    for i, (x_batch, y_batch) in enumerate(train_loader, batch_size):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = train(model, loss_fun, optimizer, x_batch, y_batch)
        losses_temp.append(loss)
    print("training loss = {}".format(sum(losses_temp)/idx)+'\n')
    losses.append(sum(losses_temp)/i)
    # Validation loss
    with torch.no_grad():
        model.eval()
        v_losses_temp = []
        for i, (x_batch, y_batch) in enumerate(val_loader, batch_size):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(x_batch)
            v_losses_temp.append(loss_fun(y_batch, y_pred).item())
        print("val validation loss = {}".format(sum(v_losses_temp)/idx))
        v_losses.append(sum(v_losses_temp)/i)

    #print(model.state_dict())
    
    if epoch%10==0:
        torch.save(model.state_dict(), model_path.format(file_specifier, file_specifier, epoch))

with h5py.File(loss_path, 'w') as hf:
    hf.create_dataset('trn_losses', data=np.array(losses))
    hf.create_dataset('v_losses', data=np.array(v_losses))

print('complete')

