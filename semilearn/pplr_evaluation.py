from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from semilearn import datasets
from semilearn.nets.resnet import resnet50part
from semilearn.evaluators import Evaluator
from semilearn.pp_utils.pp_data import transforms as T
from semilearn.pp_utils.pp_data.preprocessor import Preprocessor
from semilearn.pp_utils.myserialization_pplr import load_checkpoint, copy_state_dict

parser = argparse.ArgumentParser(description="Testing the model")
first_time = 0

def get_data(name, data_dir, height, width, batch_size, workers):
    root = data_dir

    dataset = datasets.create(name, root)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    test_loader = DataLoader(Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                                          root=dataset.images_dir, transform=test_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    print("==========\nArgs:{}\n==========".format(args))

    # dataset
    dataset, test_loader = get_data(args.dataset, args.data_dir, args.height, args.width, args.batch_size, args.workers)

    # model
    #my_net_builder = net_builder('ResNet50', False, None, is_remix=False)
    #model = my_net_builder(num_classes=1000)

    model = resnet50part(num_parts=args.part, num_classes=751)
    model.cuda()
    model = nn.DataParallel(model)

    #model = loadc(model, args.resume)
    #model = lc(model, args.resume)

    # load a checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint, model)

    # evaluate
    evaluator = Evaluator(model)
    print("Test on {}:".format(args.dataset))
    result = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, rerank=args.rerank)
    return result[1]


def start():
    global first_time
    if first_time == 0:
        # data
        parser.add_argument('-d', '--dataset', type=str, default='market1501')
        parser.add_argument('-b', '--batch-size', type=int, default=64)
        parser.add_argument('-j', '--workers', type=int, default=4)
        parser.add_argument('--height', type=int, default=256, help="input height")
        parser.add_argument('--width', type=int, default=128, help="input width")

        # path
        working_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
        parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))

        # testing configs
        parser.add_argument('--resume', type=str, metavar='PATH', default='/home/zakerian/USB/fork2/Semi-supervised-learning/saved_models/usb_cv/fixmatch_market1501_resent50part_eval_size256/latest_model.pth')
        parser.add_argument('--rerank', action='store_true', help="evaluation only")
        parser.add_argument('--seed', type=int, default=1)

        # model configs
        parser.add_argument('--part', type=int, default=3, help="number of part")
        parser.add_argument("--c", type=str, default="")

        first_time = 1

    return main()
