import numpy as np
import torch
import torch.nn as nn

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

class TDNModel(nn.Module):

