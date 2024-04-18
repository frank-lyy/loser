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

class TDNLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(TDNLoss, self).__init__()
        self.alpha_rec = 0.3
        self.gamma_rc = 0.1
        self.gamma_sm = 0.1
        self.c = 10 # TODO: have no idea what this is supposed to be
        self.device = device

    def forward(self, inputs, targets):
        reflectance_low, illumination_low, reflectance_normal, illumination_normal = inputs
        image_low, image_normal = targets

        gray_image_low = 0.299 * image_low[:, 0, :, :] + 0.587 * image_low[:, 1, :, :] + 0.114 * image_low[:, 2, :, :]
        gray_image_low = torch.unsqueeze(gray_image_low, dim=1)
        gray_image_normal = 0.299 * image_normal[:, 0, :, :] + 0.587 * image_normal[:, 1, :, :] + 0.114 * image_normal[:, 2, :, :]
        gray_image_normal = torch.unsqueeze(gray_image_normal, dim=1)

        illumination_low_extended = torch.cat((illumination_low, illumination_low, illumination_low), dim=1)
        illumination_normal_extended = torch.cat((illumination_normal, illumination_normal, illumination_normal), dim=1)

        # Calculate image gradients
        kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))
        kernel_x = kernel_x.to(self.device)
        kernel_y = torch.transpose(kernel_x, 2, 3)
        kernel_y = kernel_y.to(self.device)

        grad_image_low_x = torch.abs(F.conv2d(gray_image_low, kernel_x, stride=1, padding=1))
        grad_image_low_y = torch.abs(F.conv2d(gray_image_low, kernel_y, stride=1, padding=1))
        grad_image_normal_x = torch.abs(F.conv2d(gray_image_normal, kernel_x, stride=1, padding=1))
        grad_image_normal_y = torch.abs(F.conv2d(gray_image_normal, kernel_y, stride=1, padding=1))
        grad_illumination_low_x = torch.abs(F.conv2d(illumination_low, kernel_x, stride=1, padding=1))
        grad_illumination_low_y = torch.abs(F.conv2d(illumination_low, kernel_y, stride=1, padding=1))
        grad_illumination_normal_x = torch.abs(F.conv2d(illumination_normal, kernel_x, stride=1, padding=1))
        grad_illumination_normal_y = torch.abs(F.conv2d(illumination_normal, kernel_y, stride=1, padding=1))

        weight_image_low_x = torch.exp(-self.c * grad_image_low_x)
        weight_image_low_y = torch.exp(-self.c * grad_image_low_y)
        weight_image_normal_x = torch.exp(-self.c * grad_image_normal_x)
        weight_image_normal_y = torch.exp(-self.c * grad_image_normal_y)

        reconstruction_loss = F.l1_loss(reflectance_normal * illumination_normal_extended, image_normal) + self.alpha_rec * F.l1_loss(reflectance_low * illumination_low_extended, image_low) # TODO: add auxiliary cross product
        reflectance_consistency_loss = F.l1_loss(reflectance_normal, reflectance_low)
        illumination_smoothness_loss = torch.mean(weight_image_low_x * grad_illumination_low_x + weight_image_low_y * grad_illumination_low_y + weight_image_normal_x * grad_illumination_normal_x + weight_image_normal_y * grad_illumination_normal_y)

        # print(reconstruction_loss, reflectance_consistency_loss, illumination_smoothness_loss)

        return reconstruction_loss + self.gamma_rc * reflectance_consistency_loss + self.gamma_sm * illumination_smoothness_loss

class RetinexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 64
        self.kernel_size = 3

        self.feature_extraction = nn.Conv2d(3, self.channels, self.kernel_size * 3, padding=4, padding_mode='replicate')
        self.convolutions = nn.Sequential(nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                          nn.ReLU(),
                                          nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                          nn.ReLU(),
                                          nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                          nn.ReLU(),
                                          nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                          nn.ReLU())
        self.reconstruction = nn.Conv2d(self.channels, 4, self.kernel_size, padding=1, padding_mode='replicate')
        self.full = nn.Sequential(self.feature_extraction, self.convolutions, self.reconstruction)

    def forward(self, inputs):
        low_img, normal_img = inputs

        low_out = self.full(low_img)
        reflectance_low = torch.sigmoid(low_out[:, :3, :, :])
        illumination_low = torch.sigmoid(low_out[:, 3:, :, :])

        normal_out = self.full(normal_img)
        reflectance_normal = torch.sigmoid(normal_out[:, :3, :, :])
        illumination_normal = torch.sigmoid(normal_out[:, 3:, :, :])

        return reflectance_low, illumination_low, reflectance_normal, illumination_normal 

def train_model(device, model, dataloaders, criterion, optimizer, num_epochs, save_dir=None):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # Iterate over data, using TQDM for progress tracking
            for inputs, targets in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model((inputs, targets))
                    loss = criterion(outputs, (inputs, targets))

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
    n_epochs = 5
    lr = 0.0001
    batch_size = 16
    shuffle_datasets = True
    save_dir = "decomposition/simple_retinex_model"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RetinexModel()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = TDNLoss(device=device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = RetinexDataset('LOLdataset/train', transform=transform)
    val_dataset = RetinexDataset('LOLdataset/val', transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_datasets)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_datasets)
    dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

    model = train_model(device, model, dataloaders, criterion, optim, n_epochs, save_dir)
