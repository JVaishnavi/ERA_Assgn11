#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from model.resnet import ResNet18

batch_size = 512
kwargs = dict(shuffle=True, batch_size=batch_size) 
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

from torch.optim.lr_scheduler import OneCycleLR

num_epochs = 30
model = ResNet18().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(reduction="sum")
scheduler = OneCycleLR(optimizer,
            max_lr=6E-02,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=5/num_epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )


from tqdm import tqdm
def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


#Function to train the model and return training losses

def train(model, device, train_loader, optimizer,train_criterion):
  
    model.train()
    pbar = tqdm(train_loader)
  
    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
    
        # Predict
        pred = model(data)
    
        # Calculate loss
        loss = train_criterion(pred, target)
        train_loss+=loss.item()
    
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)
    
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    return 100. * correct / processed, train_loss

# Function to test the model and return test losses

def test(model, device, test_loader, test_criterion):


    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += test_criterion(output, target).item()*len(target)  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)


    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset), test_loss


train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


