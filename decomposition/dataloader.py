import cv2
import numpy as np

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
    import matplotlib.pyplot as plt

    # Example usage:
    directory_path = 'LOLdataset/train/low/'
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
