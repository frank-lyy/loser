import numpy as np
from dataloader import *
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn

from tqdm import tqdm
from robust_decomposition import get_noise

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    save_directory = 'decomposition/robust_noise'

    for phase in ['train', 'val']:
      directory = os.path.join('LOLdataset', phase)
      directory = os.path.join(directory, 'low')

      img_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')]
      
      for img in tqdm(img_list):
        image_low = load_png_image(os.path.join(directory, img))
        hsv_low = convert_to_hsv(image_low)
        _, _, low_v = cv2.split(hsv_low)

        low_v = torch.from_numpy(low_v)
        low_v = low_v.to(device)

        # print("starting to get noise")
        
        low_noise = get_noise(low_v)
        low_noise = low_noise.detach().numpy()

        # print("finished getting noise")
        print(low_noise)
        print(img)
    
        save_path = os.path.join(save_directory, img)
        torch.save(low_noise, save_path)

        # print("finished one image")
        break
