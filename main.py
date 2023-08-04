import numpy as np
from glob import glob
from random import shuffle, seed
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import os


class CatsDogs(Dataset):
    def __init__(self, folder):
        # Get all cat and dog image paths
        cats_folder = os.path.join('Data', 'Cats')
        dogs_folder = os.path.join('Data', 'Dogs')

        cats = glob(os.path.join(cats_folder, '*.jpg'))
        dogs = glob(os.path.join(dogs_folder, '*.jpg'))

        self.fpaths = cats + dogs
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        seed(10)
        shuffle(self.fpaths)

        # Create targets list (1 if dog, 0 if cat)
        self.targets = [fpath.split('/')[-1].startswith('dog')
                        for fpath in self.fpaths]

    def __len__(self):

        return len(self.fpaths)

    def __getitem__(self, ix):

        f = self.fpaths[ix]
        target = self.targets[ix]
        # OpenCV reads images in BGR format, so we reverse the channels to get RGB
        im = cv2.imread(f)[:, :, ::-1]
        im = cv2.resize(im, (224, 224))
        # Convert the image to a tensor and normalize pixel values to [0,1]
        im = torch.tensor(im/255)
        # Permute the image dimensions to match the PyTorch expectation (C, H, W)
        im = im.permute(2, 0, 1)
        im = self.normalize(im)  # Apply the defined normalization

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return im.float().to(device), torch.tensor([target]).float().to(device)


data = CatsDogs('../Data/train')
image, label = data[300]

image = image.permute(1, 2, 0).cpu()

plt.imshow(image)
print("Label:", label.item())  # assuming label is a tensor

# Interpret the label
if label.item() == 1:
    print("This image is a Dog.")
else:
    print("This image is a Cat.")
