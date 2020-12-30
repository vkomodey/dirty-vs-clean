import zipfile
import os
import cv2
import shutil
import numpy as np
from constants import classes_vocab
from zipfile import ZipFile
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from typing import List, Tuple
from pathlib import Path

from constants import ImageClasses

class PlatesDataset(Dataset):
    def __init__(self, images: List[List], transform): # image: (path, class)
        self.images = images
        self.transform = transform


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, cls = self.images[idx]
        img_path = str(img_path)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, classes_vocab[cls]

def get_dataset(images: List[List], transformers=None) -> DatasetFolder:
    return PlatesDataset(images, transform=transformers)

def get_train_exemplars() -> Tuple[np.array, np.array]:
    currpath = Path('.')
    dirty_path = currpath/'dataset'/'raw'/'train'/'dirty'
    cleaned_path = currpath/'dataset'/'raw'/'train'/'cleaned'
    dirty_plates_paths = list(dirty_path.glob('*'))
    cleaned_plates_paths = list(cleaned_path.glob('*'))

    train_plates = np.array(dirty_plates_paths + cleaned_plates_paths)
    train_classes = np.array(['dirty',]*len(dirty_plates_paths) + ['cleaned',]*len(cleaned_plates_paths))

    return train_plates, train_classes
