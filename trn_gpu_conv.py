import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import sys
import os

file_specifier = sys.argv[1]

if not os.path.exists('/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/{}'.format(file_specifier)):
    os.makedirs('/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/{}'.format(file_specifier))

model_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/{}/model_{}_epoch_{}.pt'
tain_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/train_{}.pt'.format(file_specifier)
val_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/val_{}.pt'.format(file_specifier)
loss_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/loss_{}.h5'.format(file_specifier)
params_path = '/cds/home/i/isele/tmox51020/results/erik/nn_reconst/models/params_{}.h5'.format(file_specifier)

data = np.load('/cds/home/i/isele/tmox51020/results/erik/nn_reconst/trn_data/run112_unstreaked.npz')
Msol1s_unstreaked = data['Msol1s_unstreaked']

trn_data = '/cds/home/i/isele/tmox51020/scratch/recon/trn_data_v9.h5'

with h5py.File(trn_data, 'r') as hf:
    x = np.array(hf.get('Msol1s_diff')[:-1000])
    x = x[:, None, :, :] # append dimension for CNN
    y = np.array(hf.get('Ets_interp_ap')[:-1000])
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

# Create pytorch tensors
x_tensor = torch.from_numpy(x).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)
# create tensor dataset from tensor
dataset = TensorDataset(x_tensor, y_tensor)

# Save model hyperparamters
params = {'weight_decay': 0, 'alpha':1e-1, 'batch_size':25, 'trn_data':trn_data, 'notes':'using convolutional neural network, with dropout of 0.2'}

# split data into train and eval
train_dataset, val_dataset = random_split(dataset, [498000, 1000])

# create data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'])
val_loader = DataLoader(dataset=val_dataset, batch_size=params['batch_size'])
      
image_size = 64

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # Code adjusted based on model
            # nn.Linear(image_size*image_size, params['L1']),
            # nn.ReLU(),
            # nn.Linear(params['L1'], 256),
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(7, 7), padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(image_size*image_size*10, 256)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork()
model.to(device)

loss_fun = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=params['alpha'], weight_decay=params['weight_decay'])

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

n_epochs = 500

losses = []
v_losses = []
for epoch in range(n_epochs):
    print(epoch)
    losses_temp = []
    for i, (x_batch, y_batch) in enumerate(train_loader, params['batch_size']):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = train(model, loss_fun, optimizer, x_batch, y_batch)
        losses_temp.append(loss)
    print("training loss = {}".format(sum(losses_temp)/i)+'\n')
    losses.append(sum(losses_temp)/i)
    # Validation loss
    with torch.no_grad():
        model.eval()
        v_losses_temp = []
        for i, (x_batch, y_batch) in enumerate(val_loader, params['batch_size']):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(x_batch)
            v_losses_temp.append(loss_fun(y_batch, y_pred).item())
        print("val validation loss = {}".format(sum(v_losses_temp)/i))
        v_losses.append(sum(v_losses_temp)/i)

    #print(model.state_dict())
    
    if epoch%10==0:
        torch.save(model.state_dict(), model_path.format(file_specifier, file_specifier, epoch))
        
# Save files

with h5py.File(loss_path, 'w') as hf:
    hf.create_dataset('trn_losses', data=np.array(losses))
    hf.create_dataset('v_losses', data=np.array(v_losses))
    
with open(params_path, 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
print('complete')

