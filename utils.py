#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:03:15 2023

@author: vaishnavijanakiraman
"""


import torch
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2

    
train_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=36, min_width=36),
        A.RandomCrop(height=32, width=32),
        
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8,
                        min_holes=1, min_height=8, min_width=8, 
                        fill_value=(0.4914, 0.4822, 0.4465)),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ToTensorV2()
])


class CIFAR10_ds(datasets.CIFAR10):
    def __init__(self, root=".", train=True, download=True, transform= None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    

train_data = CIFAR10_ds(root=".", train=True, download=True, transform=train_transforms)
test_data = CIFAR10_ds(root=".", train=False, download=True, transform=test_transforms)
    
   
#Function to plot the charts (Accuracy vs epochs)
 
def plot_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
#Function to get sumary of the neural net

from torchsummary import summary
def get_summary(Net):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))
    return summary


import numpy as np

def get_incorrect(model, test_loader):
    incorrect_examples = []
    incorrect_labels = []
    incorrect_pred = []
    incorrect_images = []
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) 
        idxs_mask = ((pred == target.view_as(pred))==False).view(-1)
        if idxs_mask.numel(): 
            incorrect_examples.append(data[idxs_mask].squeeze().cpu().numpy())
            incorrect_labels.append(target[idxs_mask].cpu().numpy())
            incorrect_pred.append(pred[idxs_mask].squeeze().cpu().numpy())
            incorrect_images.append(data[idxs_mask])
            
    return incorrect_examples, incorrect_labels, incorrect_pred, incorrect_images

def print_misclassified_images(model, test_loader, n=10):
    
    model.eval()
    incorrect_examples, incorrect_labels, incorrect_pred, incorrect_image = get_incorrect(model, test_loader)
    classes = train_data.classes
     
    fig = plt.figure(figsize=(20, 16))
    for idx in np.arange(n):
        ax = fig.add_subplot(n//5, 5, idx+1, xticks=[], yticks=[])
        img = incorrect_examples[0][idx]
        img = img/2 + 0.5
        img = np.clip(img, 0, 1)
        plt.imshow(np.transpose(incorrect_examples[0][idx], (1, 2, 0)))
        ax.set_title(f"Predicted label: {classes[incorrect_pred[0][idx]]}\n Actual label: {classes[incorrect_labels[0][idx]]}")
        
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def print_gradcam_images(model, test_loader, n=10):
    
    target_layers = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    incorrect_examples, incorrect_labels, incorrect_pred, incorrect_image = get_incorrect(model, test_loader)
    
    classes = train_data.classes
    
    fig = plt.figure(figsize=(20, 16))
    for idx in np.arange(n):
        ax = fig.add_subplot(n//5, 5, idx+1, xticks=[], yticks=[])
        input_tensor = torch.tensor(incorrect_examples[0][idx:idx+1])
        targets = [ClassifierOutputTarget(incorrect_labels[0][idx])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        rgb_img = incorrect_examples[0][idx]
        rgb_img = rgb_img/2 + 0.5
        rgb_img = np.clip(rgb_img, 0, 1)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        ax.set_title(f"Predicted label: {classes[incorrect_pred[0][idx]]}\n Actual label: {classes[incorrect_labels[0][idx]]}")
        plt.imshow(visualization)
