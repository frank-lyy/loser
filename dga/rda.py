import sys
sys.path.append('decomposition')

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import itertools
import os
import cv2

from torchvision import transforms
from tqdm import tqdm

from dataloader import RetinexDataset, load_png_image, convert_to_hsv
from tdn import RetinexModel

class ForwardDiffusionModel():
    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution

    def diffuse(self, inputs, steps=250, beta=0.02):
        samples = [inputs]
        xt = inputs
        for t in range(steps):
            q = torch.distributions.Normal(np.sqrt(1 - beta) * xt, np.sqrt(beta))
            xt = q.sample()
            samples.append(xt)

        return samples

    def sample_from_distribution(self, sample_shape):
        distribution = torch.flatten(self.distribution)
        sample = np.random.choice(distribution, size=sample_shape, replace=True)
        return torch.from_numpy(sample)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    retinex_model = RetinexModel()
    retinex_model.load_state_dict(torch.load('decomposition/simple_retinex_model/weights_last.pt', map_location=device))
    retinex_model = retinex_model.to(device)
    retinex_model.eval()
    forwardModel = ForwardDiffusionModel(torch.tensor([1, 2, 3]))

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_image = "1.png"
    low_image = convert_to_hsv(load_png_image(os.path.join('LOLdataset/test/low', test_image)))
    normal_image = convert_to_hsv(load_png_image(os.path.join('LOLdataset/test/high', test_image)))

    low_image = transform(low_image)
    low_image = torch.unsqueeze(low_image, 0)
    normal_image = transform(normal_image)
    normal_image = torch.unsqueeze(normal_image, 0)

    reflectance_low, illumination_low, reflectance_normal, illumination_normal = retinex_model((low_image, normal_image)) 
    diffused_reflectance_normal = forwardModel.diffuse(reflectance_normal)

    print(diffused_reflectance_normal[0])
    print(forwardModel.sample_from_distribution((5, 5)))
