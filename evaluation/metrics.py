import sys
sys.path.append('decomposition')
sys.path.append('dga')

import skimage.metrics
import torchmetrics
import numpy as np
import torch
import os
from generate_adjusted_reflectance import reconstruct_adjusted_reflectance
from dataloader import *
import matplotlib.pyplot as plt

def single_image_psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

def generate_psnr_plot():
    psnrs = []
    # List all files in the folder
    files = os.listdir('LOLdataset/train/high')
    # Filter for image files (assuming common image extensions like .jpg, .png, etc.)
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    # Iterate over the image files
    for image_file in image_files:
        true_image = torch.Tensor(load_png_image(os.path.join('LOLdataset/train/high/', image_file)))
        generated_image = torch.load(os.path.join('dga/reconstructed_reflectance_adjusted', image_file))
        generated_image = generated_image.squeeze(0)
        generated_image = torch.permute(generated_image, (1, 2, 0))

        psnrs.append(single_image_psnr(true_image, generated_image).detach().numpy())
                    

    # # Create histogram
    # plt.hist(psnrs, bins=10, edgecolor='black')

    # # Add labels and title
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of PSNRs')

    # # Show plot
    # plt.show()
    
    best_image_idx = np.argmax(psnrs)
    best_image_file = image_files[best_image_idx]
    print(best_image_file)
    return torch.load(os.path.join('dga/reconstructed_reflectance_adjusted', best_image_file))

def frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Calculate the Frechet distance between two multivariate normal distributions.
    """
    assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    covmean = sigma1.dot(sigma2.T)
    return np.sqrt(np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean))

def inception_score(true_images, generated_images):
    """
    Calculate the Frechet distance between the distribution of true high-light images
    and the distribution of the generated high-light images.
    """
    mu1 = np.mean(true_images, axis=0)
    sigma1 = np.cov(true_images, rowvar=False)
    mu2 = np.mean(generated_images, axis=0)
    sigma2 = np.cov(generated_images, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def generate_inception_score():
    true_image_files = os.listdir('LOLdataset/train/high')
    generated_image_files = os.listdir('dga/reconstructed_reflectance_adjusted')

    def tensor_to_image(tensor):
        image = tensor.squeeze(0)
        image = torch.permute(image, (1, 2, 0))
        return image
    
    true_images = [torch.Tensor(load_png_image(os.path.join('LOLdataset/train/high/', image_file))).detach().numpy() 
                   for image_file in true_image_files]
    generated_images = [tensor_to_image(torch.load(os.path.join('dga/reconstructed_reflectance_adjusted', image_file))).detach().numpy() 
                        for image_file in generated_image_files]

    return torchmetrics.f(true_images, generated_images)

def structural_similarity(true_images, generated_images):
    """
    Calculate the structural similarity index between the distribution of true high-light images
    and the distribution of the generated high-light images.
    """
    ssim = [skimage.metrics.structural_similarity(true_image, generated_image, channel_axis=2) for true_image, generated_image in zip(true_images, generated_images)]
    return ssim

def generate_ssim_plot():
    true_image_files = os.listdir('LOLdataset/train/high')
    generated_image_files = os.listdir('dga/reconstructed_reflectance_adjusted')

    def tensor_to_image(tensor):
        image = tensor.squeeze(0)
        image = torch.permute(image, (1, 2, 0))
        return image.to(torch.uint8)
    
    true_images = [torch.Tensor(load_png_image(os.path.join('LOLdataset/train/high/', image_file))).to(torch.uint8).detach().numpy()
                   for image_file in true_image_files]
    generated_images = [tensor_to_image(torch.load(os.path.join('dga/reconstructed_reflectance_adjusted', image_file))).detach().numpy()
                        for image_file in generated_image_files]
    
    ssim0 = structural_similarity(true_images, generated_images)

    # Create histogram
    plt.hist(ssim0, bins=10, edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of SSIs')

    # Show plot
    plt.show()


if __name__ == "__main__":
    # reconstruct_adjusted_reflectance()
    # best_image = generate_psnr_plot()
    # best_image = best_image.squeeze(0)
    # best_image = torch.permute(best_image, (1, 2, 0))
    # best_image = best_image.detach().numpy()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 1, 1)
    # plt.title('Best Image')
    # plt.imshow(best_image)
    # plt.axis('off')
    # plt.show()

    generate_ssim_plot()
