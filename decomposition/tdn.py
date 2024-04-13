import numpy as np
import torch
import torch.nn as nn

class TDNLoss(nn.Module):
    def __init__(self):
        super(TDNLoss, self).__init__()
        self.alpha_rec = 0.3
        self.gamma_rc = 0.1
        self.gamma_sm = 0.1

    def forward(self, inputs, targets):
        reflectance_normal, reflectance_low, illumination_normal, illumination_low = inputs
        image_normal, image_low = targets

        reconstruction_loss = np.linalg.norm(reflectance_normal * illumination_normal - image_normal, ord=1) + self.alpha_rec * np.linalg.norm(reflectance_low * illumination_low - image_low, ord=1)


class TDNModel(nn.Module):

