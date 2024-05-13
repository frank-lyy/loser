import cv2
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

from robust_decomposition import normalize

def check_validity(directory_path):
  images = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and not f.startswith('.')]
  invalid_images = []

  for img in images:
    image = torch.load(os.path.join(directory_path, image_filename))
    if np.min(image) == 0 and np.max(image) == 0:
      invalid_images.append(img)

  print(invalid_images)

if __name__ == "__main__":
    directory_path = 'decomposition/robust_noise'
    image_filename = '638.png'

    check_validity(directory_path)

    image = torch.load(os.path.join(directory_path, image_filename))
    image = normalize(torch.from_numpy(image), 255)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title('Noise Image')
    plt.imshow(image)
    plt.show()
