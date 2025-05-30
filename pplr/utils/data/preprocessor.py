from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, is_lb=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.is_lb = is_lb

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        try:
            fname, pid, camid = self.dataset[index]
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, fname, pid, camid, index, self.is_lb
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            return None
