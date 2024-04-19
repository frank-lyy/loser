import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from torchvision import transforms

from dataloader import load_png_image
from tdn import RetinexModel

def plot_batch_tensor(nrows, ncols, index, title, tensor):
    img = tensor[0, :, :, :]
    img = img.detach().numpy()
    img = img * 255.0
    img = np.transpose(img, [1, 2, 0])
    img = img.astype(np.uint8)

    plt.subplot(nrows, ncols, index)
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model = RetinexModel()
    best_model.load_state_dict(torch.load('decomposition/simple_retinex_model/weights_last.pt', map_location=device))
    best_model = best_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_image = "1.png"
    low_image = load_png_image(os.path.join('LOLdataset/test/low', test_image))
    normal_image = load_png_image(os.path.join('LOLdataset/test/high', test_image))

    low_image = transform(low_image)
    low_image = torch.unsqueeze(low_image, 0)
    normal_image = transform(normal_image)
    normal_image = torch.unsqueeze(normal_image, 0)

    reflectance_low, illumination_low, reflectance_normal, illumination_normal = best_model((low_image, normal_image)) 

    illumination_low_extended = torch.cat((illumination_low, illumination_low, illumination_low), dim=1)
    illumination_normal_extended = torch.cat((illumination_normal, illumination_normal, illumination_normal), dim=1)
    reconstructed_low = reflectance_low * illumination_low_extended
    reconstructed_normal = reflectance_normal * illumination_normal_extended

    plt.figure(figsize=(16, 6))
    plot_batch_tensor(4, 2, 1, 'Low Image', low_image)
    plot_batch_tensor(4, 2, 2, 'Normal Image', normal_image)
    plot_batch_tensor(4, 2, 3, 'Low Reflectance', reflectance_low)
    plot_batch_tensor(4, 2, 4, 'Low Illumination', illumination_low_extended)
    plot_batch_tensor(4, 2, 5, 'Normal Reflectance', reflectance_normal)
    plot_batch_tensor(4, 2, 6, 'Normal Illumination', illumination_normal_extended)
    plot_batch_tensor(4, 2, 7, 'Reconstructed Low', reconstructed_low)
    plot_batch_tensor(4, 2, 8, 'Reconstructed Normal', reconstructed_normal)

    plt.show()
