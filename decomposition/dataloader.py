import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

class RetinexDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.low_dir = os.path.join(self.root_dir, 'low')
        self.high_dir = os.path.join(self.root_dir, 'high')
        self.transform = transform
        self.img_list = [f for f in os.listdir(self.low_dir) if os.path.isfile(os.path.join(self.low_dir, f)) and not f.startswith('.')]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_low_name = os.path.join(self.low_dir, self.img_list[idx])
        img_high_name = os.path.join(self.high_dir, self.img_list[idx])
        print(img_low_name, img_high_name)

        img_low = load_png_image(img_low_name)
        img_high = load_png_image(img_high_name)

        if self.transform:
            img_low = self.transform(img_low)
            img_high = self.transform(img_high)

        return img_low, img_high

def load_png_image(file_path):
    """
    Load a .png image from file path.

    Parameters:
    file_path (str): Path to the .png image file.

    Returns:
    np.ndarray: Loaded image as a numpy array.
    """
    # Read image from file
    image = cv2.imread(file_path)
    
    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def convert_to_hsv(image):
    """
    Convert an image to HSV format.

    Parameters:
    image (np.ndarray): Input image as a numpy array.

    Returns:
    np.ndarray: Image converted to HSV format.
    """
    # Convert image from RGB to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    return hsv_image

if __name__ == "__main__":
    # Display the original and HSV images
    # Example usage:
    directory_path = 'LOLdataset/our485/low/'
    image_filename = '27.png'
    image = load_png_image(directory_path+image_filename)
    hsv_image = convert_to_hsv(image)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('HSV Image')
    plt.imshow(hsv_image)
    plt.axis('off')

    plt.show()
