import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2 as cv
import os

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split=0.9):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.Y = []
        for image_name in os.listdir(root_dir):
                image_path = os.path.join(root_dir, image_name)
                self.images.append(image_path)
                self.Y.append(image_name[-10:-4])
        self.images = torch.tensor(self.images)
        self.Y = torch.tensor(self.Y)
        self.split_index = int(split * len(self.images))
        self.train_images = self.images[:self.split_index]
        self.train_labels = self.Y[:self.split_index]
        self.test_images = self.images[self.split_index:]
        self.test_labels = self.Y[self.split_index:]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv.imread(self.images[idx])
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        label = self.Y[idx]
        return (img, label)
