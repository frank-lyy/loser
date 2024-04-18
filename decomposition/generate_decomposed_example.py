import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from dataloader import load_png_image

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model = torch.load('decomposition/simple_retinex_model/weight_last.pt')
    best_model = best_model.to(device)
    best_model = best_model.eval()

    test_image = "1.png"
    low_image = load_png_image(os.path.join('LOLdataset/test', test_image))
    high_image = load_png_image(os.path.join('LOLdataset/test', test_image))
