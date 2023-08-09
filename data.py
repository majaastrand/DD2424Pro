import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch


# data_folder_im = "C:\\Users\\Maja\\Documents\\Skola\\År4\\Deep\\DD2424Pro\\images"
# filename_list = []
# unique_breeds = set()

# for root, dirs, files in os.walk(data_folder_im):
#     for file in files:
#         if file.endswith('.jpg'):
#             filename_list.append(file)
#             breed_name = file.rsplit('_', 1)[0]
#             unique_breeds.add(breed_name)

# unique_breeds = list(unique_breeds)  # Convert set to list

# dog_list = []
# cat_list = []
# for breed in unique_breeds:
#     if breed[0].isupper():
#         cat_list.append(breed)
#     else:
#         dog_list.append(breed)


# data_folder = (
#     "C:\\Users\\Maja\\Documents\\Skola\\År4\Deep\\DD2424Pro\\dataset")


# base_dir = "dataset"
# os.makedirs(base_dir, exist_ok=True)

# all breeds
# all_breeds = dog_list + cat_list

# train_ratio = 0.7
# test_ratio = 0.1
# validation_ratio = 0.2

# # Move images to directories
# for breed in all_breeds:
#     breed_images = [
#         filename for filename in filename_list if breed in filename.rsplit('_', 1)[0]]
#     random.shuffle(breed_images)

#     num_images = len(breed_images)
#     num_train = int(num_images * train_ratio)
#     num_test = int(num_images * test_ratio)
#     num_validation = num_images - num_train - num_test

#     # Create breed subdirectories
#     for subfolder in ['train', 'test', 'val']:
#         path = os.path.join(base_dir, breed, subfolder)
#         os.makedirs(path, exist_ok=True)

#     train_images = breed_images[:num_train]
#     test_images = breed_images[num_train:num_train + num_test]
#     validation_images = breed_images[num_train + num_test:]

#     for image in train_images:
#         source_path = os.path.join(data_folder_im, image)
#         dest_path = os.path.join(base_dir, breed, 'train', image)
#         shutil.copy(source_path, dest_path)

#     for image in test_images:
#         source_path = os.path.join(data_folder_im, image)
#         dest_path = os.path.join(base_dir, breed, 'test', image)
#         shutil.copy(source_path, dest_path)

#     for image in validation_images:
#         source_path = os.path.join(data_folder_im, image)
#         dest_path = os.path.join(base_dir, breed, 'val', image)
#         shutil.copy(source_path, dest_path)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(
            root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


def get_breed_loaders(breed_folder):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }

    train_dataset = CustomDataset(os.path.join(
        breed_folder, 'train'), data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_dataset = CustomDataset(os.path.join(
        breed_folder, 'val'), data_transforms['val'])
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    test_dataset = CustomDataset(os.path.join(
        breed_folder, 'test'), data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    return train_loader, val_loader, test_loader


breed_folder = "C:\\Users\\Maja\\Documents\\Skola\\År4\\Deep\\DD2424Pro\\dataset\\Birman"
train_loader, val_loader, test_loader = get_breed_loaders(breed_folder)


# dataset_path = "C:\\Users\\Maja\\Documents\\Skola\\År4\\Deep\\DD2424Pro\\dataset"
# breeds = [folder for folder in os.listdir(
#     dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

# for breed in breeds:
#     breed_path = os.path.join(dataset_path, breed)
#     train_loader, val_loader, test_loader = get_breed_loaders(breed_path)
