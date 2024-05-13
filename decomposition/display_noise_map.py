import cv2
import os
import matplotlib.pyplot as plt
import torch

from robust_decomposition import normalize

if __name__ == "__main__":
    directory_path = 'decomposition/robust_noise'
    image_filename = '638.png'

    image = torch.load(os.path.join(directory_path, image_filename))
    image = normalize(torch.from_numpy(image), 255)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title('Noise Image')
    plt.imshow(image)
    plt.show()
