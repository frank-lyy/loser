import sys
sys.path.append('decomposition')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
import cv2
import torchmetrics
import os

from tqdm import tqdm
from torchvision import transforms

from dataloader import RetinexDataset
from tdn import RetinexModel

class IDALoss(nn.Module):
    def __init__(self, device='cpu'):
        super(IDALoss, self).__init__()
        self.alpha = 0.5
        self.device = device

    def forward(self, inputs, targets):
        adjusted_illumination_low, reflectance_normal, illumination_normal = inputs
        image_normal = targets

        adjusted_illumination_low_extend = torch.cat((adjusted_illumination_low, adjusted_illumination_low, adjusted_illumination_low), dim=1)

        illumination_loss = F.l1_loss(adjusted_illumination_low, illumination_normal)
        reconstruction_loss = F.l1_loss(reflectance_normal * adjusted_illumination_low, image_normal)

        return reconstruction_loss + self.alpha * illumination_loss

class IDAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 64
        self.kernel_size = 3

        self.model = nn.Sequential(nn.Conv2d(4, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                   nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                   nn.ReLU(), 
                                   nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                   nn.ReLU(), 
                                   nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                   nn.ReLU(),
                                   nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                   nn.ReLU(),
                                   nn.Conv2d(self.channels, 1, self.kernel_size, padding=1, padding_mode='replicate'))

    def forward(self, inputs):
        reflectance_low, illumination_low = inputs
        adjusted_illumination = self.model(torch.cat((reflectance_low, illumination_low), dim=1))
        return adjusted_illumination

def train_model(device, retinex_model, ida_model, dataloaders, criterion, optimizer, num_epochs, save_dir=None):
    retinex_model.eval()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                ida_model.train()
            else:
                ida_model.eval()

            running_loss = 0.0

            # Iterate over data, using TQDM for progress tracking
            for inputs, targets in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                targets = targets.to(device)
                reflectance_low, illumination_low, reflectance_normal, illumination_normal = retinex_model((inputs, targets))

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = ida_model((reflectance_low, illumination_low))
                    loss = criterion((outputs, reflectance_normal, illumination_normal), targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    # Save the model
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, 'weights_last.pt'))

    return model

if __name__ == "__main__":
    # Constants
    n_epochs = 20
    lr = 0.0001
    batch_size = 16
    shuffle_datasets = True
    save_dir = "dga/illumination_adjustment"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    retinex_model = RetinexModel()
    retinex_model.load_state_dict(torch.load('decomposition/simple_retinex_model/weights_last.pt', map_location=device))
    ida_model = IDAModel()
    ida_model = ida_model.to(device)
    optim = torch.optim.Adam(ida_model.parameters(), lr=lr)
    criterion = IDALoss(device=device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = RetinexDataset('LOLdataset/train', transform=transform)
    val_dataset = RetinexDataset('LOLdataset/val', transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_datasets)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_datasets)
    dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

    model = train_model(device, retinex_model, ida_model, dataloaders, criterion, optim, n_epochs, save_dir)
