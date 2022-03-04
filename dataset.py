import os
import csv
import lmdb
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, nerf_resolution=64):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        if random.random() > 0.5:
            img = TF.hflip(img)

        thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution), Image.HAMMING)
        img = self.transform(img)
        thumb_img = self.transform(thumb_img)

        return img, thumb_img
