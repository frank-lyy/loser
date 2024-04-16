import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import copy

from tqdm import tqdm

from scipy.signal import convolve2d as conv2d

class TDNLoss(nn.Module):
    def __init__(self):
        super(TDNLoss, self).__init__()
        self.alpha_rec = 0.3
        self.gamma_rc = 0.1
        self.gamma_sm = 0.1
        self.c = 2 # TODO: have no idea what this is supposed to be

    def forward(self, inputs, targets):
        reflectance_normal, reflectance_low, illumination_normal, illumination_low = inputs
        image_normal, image_low = targets

        # Calculate image gradients
        kernel = np.array([[-1, 0, 1]])
        weight_image_low_x = np.exp(-c * conv2d(image_low, kernel, 'same'))
        weight_image_low_y = np.exp(-c * conv2d(image_low, kernel.T, 'same') )
        weight_image_normal_x = np.exp(-c * conv2d(image_normal, kernel, 'same'))
        weight_image_normal_y = np.exp(-c * conv2d(image_normal, kernel.T, 'same'))
        grad_illumination_low_x = conv2d(reflectance_low, kernel, 'same'))
        grad_illumination_low_y = conv2d(reflectance_low, kernel.T, 'same'))
        grad_illumination_normal_x = conv2d(reflectance_normal, kernel, 'same'))
        grad_illumination_normal_y = conv2d(reflectance_normal, kernel.T, 'same'))

        reconstruction_loss = np.linalg.norm(reflectance_normal * illumination_normal - image_normal, ord=1) + self.alpha_rec * np.linalg.norm(reflectance_low * illumination_low - image_low, ord=1) # TODO: add auxiliary cross product
        reflectance_consistency_loss = np.linalg.norm(reflectance_normal - reflectance_low, ord=1)
        illumination_smoothness_loss = np.sqrt(np.linalg.norm(weight_image_low_x * grad_illumination_low_x) ** 2 + np.linalg.norm(weight_image_low_y * grad_illumination_low_y) ** 2) + np.sqrt(np.linalg.norm(weight_image_normal_x * grad_illumination_normal_x) ** 2 + np.linalg.norm(weight_image_normal_y * grad_illumination_normal_y) ** 2)

        return reconstruction_loss + gamme_rc * reflectance_consistency_loss + gamma_sm * illumination_smoothness_loss

class RetinexModel(nn.Module):
    def __init__(self):
        self.channels = 64
        self.kernel_size = 3

        self.feature_extraction = nn.Conv2d(4, self.channels, self.kernel_size * 3, padding=4, padding_mode='replicate')
        self.convolutions = nn.Sequential(nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                          nn.ReLU(),
                                          nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                          nn.ReLU(),
                                          nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                          nn.ReLU(),
                                          nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, padding_mode='replicate'),
                                          nn.ReLU())
        self.reconstruction = nn.Conv2d(self.channels, 4, self.kernel_size, padding=1, padding_mode='replicate')
        self.full = nn.Sequential(self.feature_extarction, self.convolutions, self.reconstruction)

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
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

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

if __name__ == "main":
    # Constants
    n_epochs = 5
    lr = 0.0001
    batch_size = 16
    shuffle_datasets = True
    save_dir = "decomposition/simple_retinex_model"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RetinexModel()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = TDNLoss()

    train_model(device, model, dataloaders, criterion, optimizer, n_epochs, save_dir)
