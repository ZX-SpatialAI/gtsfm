import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

class MonoDataset(Dataset):
    def __init__(self, image_dir, input_transform = None):
        self.input_transform = input_transform
        self.images = []
        self.num_references = 0
        self.num_queries = 0
        self.ground_truth = None

        self._load_images(image_dir)


    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def _load_images(self, image_dir):
        suffixes = ['.jpg', '.JPG', 'jpeg', 'JPEG', '.png', 'PNG']
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                suffix = os.path.splitext(file)[1]
                if suffix not in suffixes:
                    continue
                file_path = os.path.realpath(os.path.join(root, file))
                self.images.append(file_path)