import numpy as np
import os
import os.path as osp
import glob
import re
import copy
import torchvision.transforms as transforms
from semilearn.datasets.utils import split_ssl_data
from semilearn.datasets.augmentation.randaugment import RandAugment
from .datasetbase import BasicDataset  # Ensure this import is inside functions if needed
from sklearn.model_selection import train_test_split


def get_data(data_dir, train=True):
    """
    Loads the Market1501 dataset.
    """
    market1501_dir = os.path.join(data_dir, 'Market-1501-v15.09.15')
    train_dir = os.path.join(market1501_dir, 'bounding_box_train')
    gallery_dir = os.path.join(market1501_dir, 'bounding_box_test')
    query_dir = os.path.join(market1501_dir, 'query')

    train_images, train_labels = load_market1501_images(train_dir, relabel=True)
    gallery_images, gallery_labels = load_market1501_images(gallery_dir, relabel=True)
    query_images, query_labels = load_market1501_images(query_dir, relabel=True)

    test_images = np.concatenate((query_images, gallery_images))
    test_labels = np.concatenate((gallery_labels, query_labels))

    print(f"Train set: {len(train_images)} images, {len(set(train_labels))} identities")
    print(f"Test set: {len(test_images)} images, {len(set(test_labels))} identities")

    return (train_images, train_labels) if train else (test_images, test_labels)


def load_market1501_images(directory, relabel=False):
    """
    Custom function to load Market1501 dataset images and labels.
    """
    images, labels = [], []
    img_paths = glob.glob(osp.join(directory, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = {int(pattern.search(img_path).group(1)) for img_path in img_paths if pattern.search(img_path)}
    pid_container.discard(-1)  # Ignore junk images
    pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))} if relabel else None

    for img_path in img_paths:
        match = pattern.search(img_path)
        if match:
            pid, _ = map(int, match.groups())
            if pid == -1:
                continue  # Ignore junk images
            labels.append(pid2label[pid] if relabel else pid)
            images.append(img_path)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[y] for y in labels])

    return images, labels


def get_transforms():
    """
    Returns a dictionary of transformations.
    """
    train_transform = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((256,128), padding=(int(256 * (1 - 0.875)), int(128 * (1 - 0.875))), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    strong_transform = copy.deepcopy(train_transform)
    strong_transform.transforms.insert(0, RandAugment(3, 5))

    val_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, strong_transform, val_transform


def get_dset(algorithm, data_dir, num_classes, train=True, is_ulb=False, strong_transform=None, onehot=False):
    """
    Returns a dataset instance.
    """
    data, targets = get_data(data_dir, train)
    train_transform, _, _ = get_transforms()

    return BasicDataset(algorithm, data, targets, num_classes, train_transform, is_ulb, strong_transform=strong_transform, onehot=onehot)


def get_ssl_dset(args, algorithm, data_dir, num_classes, num_labels, include_lb_to_ulb=True, onehot=False):
    """
    Splits data into labeled and unlabeled sets and returns respective datasets.
    """
    data, targets = get_data(data_dir, train=True)
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
        args=args,
        num_classes=num_classes, lb_num_labels=num_labels, include_lb_to_ulb=include_lb_to_ulb,
        data=data, targets=targets
    )
    train_transform, strong_transform, _ = get_transforms()

    # lb_dset = BasicDataset(algorithm, lb_data, lb_targets, num_classes, train_transform, False, onehot=onehot)
    # ulb_dset = BasicDataset(algorithm, ulb_data, ulb_targets, num_classes, train_transform, True, strong_transform=strong_transform, onehot=onehot)
    return lb_data, lb_targets, ulb_data, ulb_targets


def get_market1501(args, algorithm, data_dir, num_classes, num_labels):
    """
    Returns labeled, unlabeled, and evaluation datasets.
    """
    lb_dset, ulb_dset = get_ssl_dset(args, algorithm, data_dir, num_classes, num_labels)
    eval_dset = get_dset(algorithm, data_dir, num_classes, train=False)
    return lb_dset, ulb_dset, eval_dset
