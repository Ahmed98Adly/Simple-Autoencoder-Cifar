# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:26:31 2021

@author: AA
"""

#import libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable


#CIFAR dataset
train_set = torchvision.datasets.CIFAR10(root= './data', 
                                            train = True,
                                              download = True,
                                              transform = transforms.Compose([transforms.ToTensor()]))

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                              transform= transforms.Compose([transforms.ToTensor()]))



train_loader = torch.utils.data.DataLoader(train_set, batch_size= 10)

test_loader = torch.utils.data.DataLoader(test_set, batch_size= 4)

#size assertion
assert len(train_set) == 50000
assert len(test_set) == 10000

#model working directory
if not os.getcwd() == 'C:/Users/AA/Desktop/En-De':
    os.chdir('C:/Users/AA/Desktop/En-De')
else:
    pass

#import autoencoder
from modelAE import Autoencoder

#parameters
epoch = 10
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)


#train
for i in range(epoch):
    for data in train_loader:
        img, _ = data
        img = Variable(img) 
        output = model(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch: {i}  loss: {loss}')

print('Finish training')

#save state_dict
PATH = 'C:/Users/AA/Desktop/En-De/state_dict'
torch.save(model.state_dict(), PATH)

#evaluation on test set by torch.no_grad or eval()
with torch.no_grad():
    for data in test_loader:
        images,_ = data
        outputs = model(images)
        loss = criterion(output, img)
    print(f'testing loss: {loss}')











