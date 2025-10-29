import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms


data_dir = "data/cell_images"
parasitized_dir = os.path.join(data_dir, "Parasitized")
uninfected_dir = os.path.join(data_dir, "Uninfected")

image_size = (128, 128)
images = []
labels = []

def load_images_from_folder(folder, label):
    for img_name in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(image_size)
            images.append(np.array(img))
            labels.append(label)
        except:
            continue

load_images_from_folder(parasitized_dir, 1)
load_images_from_folder(uninfected_dir, 0)

print(f"Total images loaded: {len(images)}")
print(f"Labels count: {np.bincount(labels)}")
