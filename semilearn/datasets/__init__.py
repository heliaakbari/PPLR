# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings
from semilearn.datasets.utils import split_ssl_data, get_collactor
from semilearn.datasets.cv_datasets import get_cifar, get_eurosat, get_imagenet, get_medmnist, get_semi_aves, get_stl10, get_svhn, get_food101, get_market1501
from semilearn.datasets.nlp_datasets import get_json_dset
from semilearn.datasets.audio_datasets import get_pkl_dset
from semilearn.datasets.samplers import name2sampler, DistributedSampler, WeightedDistributedSampler, ImageNetDistributedSampler
from .market1501 import Market1501


__factory = {
    'market1501': Market1501,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)