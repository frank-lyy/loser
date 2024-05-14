import sys
sys.path.append('decomposition')

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from torchvision import transforms

from dataloader import load_png_image, convert_to_hsv
from tdn import RetinexModel
from ida import IDAModel
from rda import ReverseDiffusionModel 

def plot_batch_tensor(nrows, ncols, index, title, tensor):
    transform = transforms.ToPILImage(mode='HSV')
    img = tensor[0, :, :, :]
    img = transform(img)

    plt.subplot(nrows, ncols, index)
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')

def plot_batch_illumination_tensor(nrows, ncols, index, title, tensor):
    img = tensor[0, 0, :, :]
    img = img.detach().numpy()
    img = img * 255

    plt.subplot(nrows, ncols, index)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

def reconstruct_adjusted_reflectance():
    """
    Generate the reconstructed image (using adjusted reflectance) for all images in the test dataset
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model = RetinexModel()
    best_model.load_state_dict(torch.load('decomposition/simple_retinex_model/weights_last.pt', map_location=device))
    best_model = best_model.eval()
    reflectance_model = ReverseDiffusionModel()
    reflectance_model.load_state_dict(torch.load('dga/reflectance_adjustment/weights_0.pt', map_location=device))
    reflectance_model = reflectance_model.eval()
    illumination_model = IDAModel()
    illumination_model.load_state_dict(torch.load('dga/illumination_adjustment/weights_last.pt', map_location=device))
    illumination_model = illumination_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # List all files in the folder
    files = os.listdir('LOLdataset/train/low')
    # Filter for image files (assuming common image extensions like .jpg, .png, etc.)
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    # Iterate over the image files
    for image_file in image_files:
        test_image = image_file
        low_image = convert_to_hsv(load_png_image(os.path.join('LOLdataset/train/low', test_image)))
        normal_image = convert_to_hsv(load_png_image(os.path.join('LOLdataset/train/high', test_image)))

        low_image = transform(low_image)
        low_image = torch.unsqueeze(low_image, 0)
        normal_image = transform(normal_image)
        normal_image = torch.unsqueeze(normal_image, 0)

        reflectance_low, illumination_low, reflectance_normal, illumination_normal = best_model((low_image, normal_image)) 
        adjusted_reflectance = reflectance_model(reflectance_low)
        adjusted_illumination = illumination_model((reflectance_low, illumination_low))

        illumination_low_extended = torch.cat((illumination_low, illumination_low, illumination_low), dim=1)
        illumination_normal_extended = torch.cat((illumination_normal, illumination_normal, illumination_normal), dim=1)
        adjusted_illumination_extended = torch.cat((adjusted_illumination, adjusted_illumination, adjusted_illumination), dim=1)
        reconstructed_low = reflectance_low * illumination_low_extended
        reconstructed_normal = reflectance_normal * illumination_normal_extended
        reconstructed_adjusted = adjusted_reflectance * adjusted_illumination_extended
        
        image_filename = os.path.join('dga/reconstructed_reflectance_adjusted', image_file)
        torch.save(reconstructed_adjusted, image_filename)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model = RetinexModel()
    best_model.load_state_dict(torch.load('decomposition/simple_retinex_model/weights_last.pt', map_location=device))
    best_model = best_model.eval()
    reflectance_model = ReverseDiffusionModel()
    reflectance_model.load_state_dict(torch.load('dga/reflectance_adjustment/weights_last.pt', map_location=device))
    reflectance_model = reflectance_model.eval()
    illumination_model = IDAModel()
    illumination_model.load_state_dict(torch.load('dga/illumination_adjustment/weights_last.pt', map_location=device))
    illumination_model = illumination_model.eval()

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

    reflectance_low, illumination_low, reflectance_normal, illumination_normal = best_model((low_image, normal_image)) 
    adjusted_reflectance = reflectance_model(reflectance_low)
    adjusted_illumination = illumination_model((reflectance_low, illumination_low))

    illumination_low_extended = torch.cat((illumination_low, illumination_low, illumination_low), dim=1)
    illumination_normal_extended = torch.cat((illumination_normal, illumination_normal, illumination_normal), dim=1)
    adjusted_illumination_extended = torch.cat((adjusted_illumination, adjusted_illumination, adjusted_illumination), dim=1)
    reconstructed_low = reflectance_low * illumination_low_extended
    reconstructed_normal = reflectance_normal * illumination_normal_extended
    reconstructed_adjusted = adjusted_reflectance * adjusted_illumination_extended

    plt.figure(figsize=(12, 6))
    plot_batch_tensor(3, 4, 1, 'Low Image', low_image)
    plot_batch_tensor(3, 4, 2, 'Low Reflectance', reflectance_low)
    plot_batch_illumination_tensor(3, 4, 3, 'Low Illumination', illumination_low)
    plot_batch_tensor(3, 4, 4, 'Reconstructed Low', reconstructed_low)
    plot_batch_tensor(3, 4, 5, 'Normal Image', normal_image)
    plot_batch_tensor(3, 4, 6, 'Normal Reflectance', reflectance_normal)
    plot_batch_illumination_tensor(3, 4, 7, 'Normal Illumination', illumination_normal)
    plot_batch_tensor(3, 4, 8, 'Reconstructed Normal', reconstructed_normal)
    plot_batch_tensor(3, 4, 10, 'Adjusted Reflectance', adjusted_reflectance)
    plot_batch_illumination_tensor(3, 4, 11, 'Adjusted Illumination', adjusted_illumination)
    plot_batch_tensor(3, 4, 12, 'Reconstructed from Low/Adjusted', reconstructed_adjusted)

    plt.show()
