import sys
sys.path.append('decomposition')

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
import itertools
import os
import cv2

from torchvision import transforms
from tqdm import tqdm

from dataloader import NoiseDataset, load_png_image, convert_to_hsv
from tdn import RetinexModel
from robust_decomposition import RobustRetinexModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class ReverseLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(ReverseLoss, self).__init__()
        self.device = device

    def forward(self, inputs, targets):
        denoised_reflectance = inputs
        original_reflectance = targets

        loss = F.l1_loss(denoised_reflectance, original_reflectance)

        return loss

class ForwardDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        reflectance_normal, distribution = inputs
        outputs = []
        for i in range(distribution.shape[0]):
            outputs.append(self.diffuse(reflectance_normal[i], distribution[i]))
        return torch.stack(outputs)

    def diffuse(self, inputs, distribution, steps=250, beta=0.02):
        xt = inputs
        for t in range(steps):
            sample = self.sample_from_distribution(distribution, xt.shape)
            xt = np.sqrt(1 - beta) * xt + np.sqrt(beta) * sample.to(device)

        return xt

    def sample_from_distribution(self, distribution, sample_shape):
        distribution = torch.flatten(distribution)
        sample = np.random.choice(distribution.cpu().detach().numpy(), size=sample_shape, replace=True)
        return torch.from_numpy(sample)

class ReverseDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 3

        # Encoder
        self.encoder_layer1 = nn.Sequential(nn.Conv2d(3, 64, self.kernel_size, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 64, self.kernel_size, padding=1),
                                            nn.ReLU())
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_layer2 = nn.Sequential(nn.Conv2d(64, 128, self.kernel_size, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, self.kernel_size, padding=1),
                                            nn.ReLU())
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_layer3 = nn.Sequential(nn.Conv2d(128, 256, self.kernel_size, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(256, 256, self.kernel_size, padding=1),
                                            nn.ReLU())
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_layer4 = nn.Sequential(nn.Conv2d(256, 512, self.kernel_size, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(512, 512, self.kernel_size, padding=1),
                                            nn.ReLU())

        # Decoder
        self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_layer1 = nn.Sequential(nn.Conv2d(512, 256, self.kernel_size, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(256, 256, self.kernel_size, padding=1),
                                            nn.ReLU())

        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_layer2 = nn.Sequential(nn.Conv2d(256, 128, self.kernel_size, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(128, 128, self.kernel_size, padding=1),
                                            nn.ReLU())

        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_layer3 = nn.Sequential(nn.Conv2d(128, 64, self.kernel_size, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 64, self.kernel_size, padding=1),
                                            nn.ReLU())

        self.decoder_layer4 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, inputs):
        # Encoder
        encode_layer1 = self.encoder_layer1(inputs)
        encode_pool1 = self.encoder_pool1(encode_layer1)
        encode_layer2 = self.encoder_layer2(encode_pool1)
        encode_pool2 = self.encoder_pool2(encode_layer2)
        encode_layer3 = self.encoder_layer3(encode_pool2)
        encode_pool3 = self.encoder_pool3(encode_layer3)
        encode_layer4 = self.encoder_layer4(encode_pool3)

        # Decoder
        decode_conv1 = self.decoder_conv1(encode_layer4)
        decode_up1 = torch.cat([decode_conv1, encode_layer3], dim=1)
        decode_layer1 = self.decoder_layer1(decode_up1)
        decode_conv2 = self.decoder_conv2(decode_layer1)
        decode_up2 = torch.cat([decode_conv2, encode_layer2], dim=1)
        decode_layer2 = self.decoder_layer2(decode_up2)
        decode_conv3 = self.decoder_conv3(decode_layer2)
        decode_up3 = torch.cat([decode_conv3, encode_layer1], dim=1)
        decode_layer3 = self.decoder_layer3(decode_up3)
        decode_layer4 = self.decoder_layer4(decode_layer3)

        return decode_layer4

def train_model(device, retinex_model, rda_forward_model, rda_reverse_model, dataloaders, criterion, optimizer, num_epochs, save_dir=None):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        

        for phase in ['train', 'val']:
            if phase == 'train':
                rda_reverse_model.train()
            else:
                rda_reverse_model.eval()

            running_loss = 0.0

            # Iterate over data, using TQDM for progress tracking
            for inputs, noise, targets in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                noise = noise.to(device)
                targets = targets.to(device)

                _, _, reflectance_normal, _ = retinex_model((inputs, targets))
                noisy_reflectance_normal = rda_forward_model.diffuse(reflectance_normal, noise)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = rda_reverse_model(noisy_reflectance_normal)
                    loss = criterion(outputs, reflectance_normal)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    if save_dir:
        torch.save(rda_reverse_model.state_dict(), os.path.join(save_dir, 'weights_last.pt'))

    return rda_reverse_model


if __name__ == "__main__":
    # Constants
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    # print(r, a)

    n_epochs = 20
    lr = 0.0001
    batch_size = 4
    shuffle_datasets = True
    save_dir = "dga/reflectance_adjustment"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    retinex_model = RetinexModel()
    retinex_model.load_state_dict(torch.load('decomposition/simple_retinex_model/weights_last.pt', map_location=device))
    retinex_model = retinex_model.to(device)
    retinex_model.eval()

    rda_forward_model = ForwardDiffusionModel()

    rda_reverse_model = ReverseDiffusionModel()
    rda_reverse_model = rda_reverse_model.to(device)

    optim = torch.optim.Adam(rda_reverse_model.parameters(), lr=lr)
    criterion = ReverseLoss(device=device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    # print(r, a)

    train_dataset = NoiseDataset('LOLdataset/train', transform=transform)
    val_dataset = NoiseDataset('LOLdataset/val', transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_datasets)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_datasets)
    dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    # print(r, a)

    model = train_model(device, retinex_model, rda_forward_model, rda_reverse_model, dataloaders, criterion, optim, n_epochs, save_dir)
