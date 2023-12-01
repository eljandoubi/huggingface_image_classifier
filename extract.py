"""
the main script that extracts images from raw files
Author: Abdelkarim eljandoubi
date: Nov 2023
"""

import numpy as np
from PIL import Image
import os
from tqdm import tqdm


def extract():
    """extract data from raw files"""
    
    # parameters
    width, height, channels = 56, 56, 3
    
    num_images = {
        "train" : 111430,
        "test" : 10130
        }
    
    save_directory = 'images'  # save folder
    
    if os.path.isdir(save_directory):
        return
    
    # create the save folder if it does not exist
    os.makedirs(save_directory, exist_ok=True)
    
    for split in ["train","test"]:
        file_path = f'db_{split}.raw'  # path to the binary file
        
        
        # create the save folders if it does not exist
        os.makedirs(f'{save_directory}/{split}', exist_ok=True)
        os.makedirs(f'{save_directory}/{split}/0', exist_ok=True)
        os.makedirs(f'{save_directory}/{split}/1', exist_ok=True)
        
        # try to find the labels file
        try:
            
            with open(f'label_{split}.txt',"r") as f:
                labels=f.readlines()
                
                
        except FileNotFoundError:
            
            labels = []
            
        # get the label without "\n"
        labels = map(lambda x:x[0], labels)
            
    
        # open the binary file
        with open(file_path, 'rb') as file:
            for i in tqdm(range(num_images[split])):
                # read image data
                image_data = file.read(width * height * channels)
                
                # check the data
                if len(image_data) != width * height * channels:
                    break
        
                # cast to numpy array
                image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))
                
                
                # get the label if any, else set it to 0
                label = next(labels,0)
        
                # cast to PIL image
                img = Image.fromarray(image, 'RGB')
                img.save(f'{save_directory}/{split}/{label}/image_{i+1}.png')
