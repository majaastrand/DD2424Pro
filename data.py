import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch
import shutil
import random

# data_folder_im = "C:\\Users\\Maja\\Documents\\Skola\\År4\\Deep\\DD2424Pro\\images"
# filename_list = []
# unique_breeds = set()

# for root, dirs, files in os.walk(data_folder_im):
#     for file in files:
#         if file.endswith('.jpg'):
#             filename_list.append(file)
#             breed_name = file.rsplit('_', 1)[0]
#             unique_breeds.add(breed_name)

# unique_breeds = list(unique_breeds)

# dog_list = []
# cat_list = []
# for breed in unique_breeds:
#     if breed[0].isupper():
#         cat_list.append(breed)
#     else:
#         dog_list.append(breed)

# data_folder = "C:\\Users\\Maja\\Documents\\Skola\\År4\\Deep\\DD2424Pro\\dataset"
# base_dir = "dataset"
# os.makedirs(base_dir, exist_ok=True)

# all_breeds = dog_list + cat_list

# train_ratio = 0.7
# test_ratio = 0.1
# validation_ratio = 0.2

# for breed in all_breeds:
#     breed_images = [
#         filename for filename in filename_list if breed in filename.rsplit('_', 1)[0]]
#     random.shuffle(breed_images)

#     num_images = len(breed_images)
#     num_train = int(num_images * train_ratio)
#     num_test = int(num_images * test_ratio)
#     num_validation = num_images - num_train - num_test

#     train_images = breed_images[:num_train]
#     test_images = breed_images[num_train:num_train + num_test]
#     validation_images = breed_images[num_train + num_test:]

#     for subfolder, images in zip(['train', 'test', 'val'], [train_images, test_images, validation_images]):
#         path = os.path.join(base_dir, subfolder, breed)
#         os.makedirs(path, exist_ok=True)

#         for image in images:
#             source_path = os.path.join(data_folder_im, image)
#             dest_path = os.path.join(base_dir, subfolder, breed, image)
#             shutil.copy(source_path, dest_path)


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'C:\\Users\\Maja\\Documents\\Skola\\År4\\Deep\\DD2424Pro\\dataset'
train_datesets = datasets.ImageFolder(os.path.join(
    data_dir, 'train'), data_transforms['train'])
train_loaders = torch.utils.data.DataLoader(
    train_datesets, batch_size=4, shuffle=True, num_workers=4)
train_dataset_sizes = len(train_datesets)

val_datesets = datasets.ImageFolder(
    os.path.join(data_dir, 'val'), data_transforms['val'])
val_loaders = torch.utils.data.DataLoader(
    val_datesets, batch_size=4, shuffle=True, num_workers=4)
val_dataset_sizes = len(val_datesets)

test_datesets = datasets.ImageFolder(
    os.path.join(data_dir, 'test'), data_transforms['test'])
test_loaders = torch.utils.data.DataLoader(
    test_datesets, batch_size=4, shuffle=True, num_workers=4)
test_dataset_sizes = len(test_datesets)

class_name = train_datesets.classes
# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
