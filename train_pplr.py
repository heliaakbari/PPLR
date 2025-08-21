from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
from re import X
import numpy as np
import sys
import time
from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from semilearn.core.evaluate_label import Algorithm
from semilearn.datasets.cv_datasets.market1501 import get_ssl_dset
from semilearn.algorithms.utils import str2bool
from semilearn.algorithms.hooks.pseudo_label import PseudoLabelingHook
from pplr import datasets
from pplr.models import resnet50part
from pplr.trainers import PPLRTrainer
from pplr.evaluators import Evaluator, extract_all_features, return_batch_images
from pplr.utils.data import IterLoader
from pplr.utils.data import transforms as T
from pplr.utils.data.sampler import RandomMultipleGallerySampler
from pplr.utils.data.preprocessor import Preprocessor
from pplr.utils.logging import Logger
from pplr.utils.faiss_rerank import compute_ranked_list, compute_jaccard_distance
from pplr.utils.myserialization_pplr import load_checkpoint, copy_state_dict
best_mAP = 0

from torchsummary import summary

def get_data(name, data_dir):
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset

def map_paths_to_full_entries(paths, full_dataset):
    """
    Given a list of image paths and a full dataset of (path, pid, camid) tuples,
    map each path back to its full tuple.
    
    Args:
        paths (list): List of image paths.
        full_dataset (list): List of (path, pid, camid) tuples.

    Returns:
        List of (path, pid, camid) tuples corresponding to the given paths.
    """
    path_to_tuple = {entry[0]: entry for entry in full_dataset}
    mapped = [path_to_tuple[p] for p in paths if p in path_to_tuple]
    return mapped


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, is_lb=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, is_lb=is_lb),
                           batch_size=batch_size, num_workers=workers, sampler=sampler,
                           shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None, is_lb=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer, is_lb=is_lb),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)   

    return test_loader


def compute_pseudo_labels(features, cluster, k1):
    mat_dist = compute_jaccard_distance(features, k1=k1, k2=6)
    ids = cluster.fit_predict(mat_dist)
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)

    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num_ids + outliers)
            outliers += 1

    return torch.Tensor(labels).long().detach(), num_ids


def compute_cross_agreement(features_g, features_p, k, search_option=0):
    print("Compute cross agreement score...")
    N, D, P = features_p.size()
    score = torch.FloatTensor()
    end = time.time()
    ranked_list_g = compute_ranked_list(features_g, k=k, search_option=search_option, verbose=False)

    for i in range(P):
        ranked_list_p_i = compute_ranked_list(features_p[:, :, i], k=k, search_option=search_option, verbose=False)
        intersect_i = torch.FloatTensor(
            [len(np.intersect1d(ranked_list_g[j], ranked_list_p_i[j])) for j in range(N)])
        union_i = torch.FloatTensor(
            [len(np.union1d(ranked_list_g[j], ranked_list_p_i[j])) for j in range(N)])
        score_i = intersect_i / union_i
        score = torch.cat([score, score_i.unsqueeze(1)], dim=1)

    print("Cross agreement score time cost: {}".format(time.time() - end))
    return score



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

    main_worker(args)


def main_worker(args):
    global best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    # dataset

    dataset = get_data(args.dataset, args.data_dir)
    print(f"dataset train-168: {len(dataset.train)}, {dataset.train[0]}")
    lb_data, lb_targets, ulb_data, ulb_target = get_ssl_dset(args, args.algorithm, args.data_dir, args.num_classes, args.num_labels)
    print(f"lb_data: {len(lb_data)}, {lb_data[0]}")
    print(f"lb_targets: {len(lb_targets)}, {lb_targets[0]} ")
    #print(', '.join(str(x) for x in lb_targets))
    print(f"ulb_data: {len(ulb_data)}, {ulb_data[0]}")
    print(f"ulb_target: {len(ulb_target)}, {ulb_target[0]}")
    #print(', '.join(str(x) for x in ulb_target))

    lb_data_dset = map_paths_to_full_entries(lb_data, dataset.train)
    ulb_data_dset = map_paths_to_full_entries(ulb_data, dataset.train)
    print(f"lb_data_dset: {len(lb_data_dset)}, {lb_data_dset[0:10]}")
    print(f"ulb_data_dset: {len(ulb_data_dset)}, {ulb_data_dset[0:10]}")


    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers, is_lb=0)
    lb_cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers,testset=sorted(lb_data_dset), is_lb=1)

    ulb_cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size*args.uratio, args.workers,testset=sorted(ulb_data_dset), is_lb=0)


    # model
    num_part = args.part
    model = resnet50part(num_parts=args.part, num_classes=751)
    model.cuda()

    summary(model, input_size=(3, 384, 128))
    print("hello")
    model = nn.DataParallel(model)


    # load a checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint, model)

    # evaluator
    evaluator = Evaluator(model)

    # optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    score_log = torch.FloatTensor([])
    for epoch in range(args.epochs):
        
        lb_features_g, lb_features_p, _, lb_labels = extract_all_features(model, lb_cluster_loader)
        lb_features_g = torch.cat([lb_features_g[f].unsqueeze(0) for f, _, _ in sorted(lb_data_dset)], 0)
        lb_features_p = torch.cat([lb_features_p[f].unsqueeze(0) for f, _, _ in sorted(lb_data_dset)], 0)
        lb_labels = torch.cat([torch.tensor([lb_labels[f]]) for f, _, _ in sorted(lb_data_dset)], dim=0)
        print(f"classes shape:{lb_labels.shape}")
        print(f"labeled features g shape:{lb_features_g.shape}")


        ulb_features_g, ulb_features_p, ulb_classes, _ = extract_all_features(model, ulb_cluster_loader)
        ulb_features_g = torch.cat([ulb_features_g[f].unsqueeze(0) for f, _, _ in sorted(ulb_data_dset)], 0)
        ulb_features_p = torch.cat([ulb_features_p[f].unsqueeze(0) for f, _, _ in sorted(ulb_data_dset)], 0)
        ulb_classes = torch.cat([torch.tensor([ulb_classes[f]]) for f, _, _ in sorted(ulb_data_dset)], dim=0)
        # print(f"cluster loader:{cluster_loader.keys()}")
        #if epoch == 0:
            #cluster = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=8)
        #    algorithm = Algorithm(model=model, x_lb=images, dataset = dataset)
        # assign pseudo-labels
        #pseudo_labling_hook = PseudoLabelingHook()
        #probs_x_ulb_w = torch.softmax(ulb_logits.detach(), dim=-1)
        pseudo_labels = ulb_classes
        num_class = 751


        # Compute the cross-agreement
        
        score = compute_cross_agreement(ulb_features_g, ulb_features_p, k=args.k)
        print(f"cross agreement score head: {score[:5]}")
        print(f"cross agreement score shape: {score.shape}")
        score_log = torch.cat([score_log, score.unsqueeze(0)], dim=0)

        # generate new dataset with pseudo-labels
        num_outliers = 0
        lb_new_dataset = []
        ulb_new_dataset = []

        idxs, pids = [], []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(ulb_data_dset), pseudo_labels)):
            pid = label.item()
            if pid >= num_class:  # append data except outliers
                num_outliers += 1
            else:
                ulb_new_dataset.append((fname, pid, cid))
                idxs.append(i)
                pids.append(pid)
        
        for j, ((lb_fname, _, lb_cid), lb_label) in enumerate(zip(sorted(lb_data_dset), lb_labels)):
            lb_pid = lb_label.item()
            if lb_pid >= num_class:  # append data except outliers
                num_outliers += 1
            else:
                lb_new_dataset.append((lb_fname, lb_pid, lb_cid))

        lb_train_loader = get_train_loader(dataset, args.height, args.width, args.batch_size,
                                        args.workers, args.num_instances, args.iters, trainset=lb_new_dataset, is_lb=1)

        ulb_train_loader = get_train_loader(dataset, args.height, args.width, args.batch_size*args.uratio,
                                        args.workers, args.num_instances, args.iters, trainset=ulb_new_dataset, is_lb=0)

        # statistics of clusters and un-clustered instances
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'.format(epoch, num_class,
                                                                                           num_outliers))

        # reindex
        idxs, pids = np.asarray(idxs), np.asarray(pids)
        #ulb_features_g = ulb_features_g[idxs, :]
        #ulb_features_p = ulb_features_p[idxs, :, :]
        score = score[idxs, :]

        print(f"idxs: {idxs.shape}, {idxs[0:10]}")
        print(f"pids: {pids.shape}, {pids[0:10]}")
        print(f"score: {score.shape}, {score[0:10]}")
        # compute cluster centroids
        centroids_g, centroids_p = [], []
        for pid in sorted(np.unique(pids)):  # loop all pids
            idxs_p = np.where(pids == pid)[0]
            centroids_g.append(ulb_features_g[idxs_p].mean(0))
            centroids_p.append(ulb_features_p[idxs_p].mean(0))

        print(f"centeriod g before normal: {len(centroids_g)}, {centroids_g[0:10]}")
        centroids_g = F.normalize(torch.stack(centroids_g), p=2, dim=1)
        print(f"centeroid g after normal: {len(centroids_g)}, {centroids_g[0:10]}")

        model.module.classifier.weight.data[:num_class].copy_(centroids_g)

        for i in range(num_part):
            centroids_p_i = torch.stack(centroids_p)[:, :, i]
            centroids_p_i = F.normalize(centroids_p_i, p=2, dim=1)
            print(f"centeroid p_{i} after normal: {len(centroids_p_i)}, {centroids_p_i[0:10]}")
            classifier_p_i = getattr(model.module, 'classifier' + str(i))
            classifier_p_i.weight.data[:num_class].copy_(centroids_p_i)

        # training
        trainer = PPLRTrainer(model, score, num_class=num_class, num_part=num_part, beta=args.beta,
                              aals_epoch=args.aals_epoch)

        print(f"len iters: {len(lb_train_loader)} and {len(ulb_train_loader)}")

        trainer.train(epoch, lb_train_loader, ulb_train_loader, optimizer=optimizer, print_freq=args.print_freq, train_iters=len(lb_train_loader))

        lr_scheduler.step()

        # evaluation
        if ((epoch+1) % args.eval_step == 0) or (epoch == args.epochs-1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,epoch, cmc_flag=False)

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), osp.join(args.logs_dir, 'best.pth'))
            print('\n* Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}\n'.format(epoch, mAP, best_mAP))

    torch.save(model.state_dict(), osp.join(args.logs_dir, 'last.pth'))
    np.save(osp.join(args.logs_dir, 'scores.npy'), score_log.numpy())

    # results
    model.load_state_dict(torch.load(osp.join(args.logs_dir, 'best.pth')))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, epoch, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Part-based Pseudo Label Refinement")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-n', '--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/test'))
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--labled-in-centroid', type=bool, default=True)
    # PPLR
    parser.add_argument('--part', type=int, default=3, help="number of part")
    parser.add_argument('--k', type=int, default=20,
                        help="hyperparameter for cross agreement score")
    parser.add_argument('--beta', type=float, default=0.5,
                        help="weighting parameter for part-guided label refinement")
    parser.add_argument('--aals-epoch', type=int, default=5,
                        help="starting epoch for agreement-aware label smoothing")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.000175, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--step-size', type=int, default=20)

    # cluster
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--eps', type=float, default=0.5,
                        help="distance threshold for DBSCAN")

    # fixmatch
    parser.add_argument(
        "--include_lb_to_ulb",
        type=str2bool,
        default="True",
        help="flag of including labeled data into unlabeled data, default to True",
    )

    ## imbalanced setting arguments
    parser.add_argument(
        "--lb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of labeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of unlabeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_num_labels",
        type=int,
        default=None,
        help="number of labels for unlabeled data, used for determining the maximum "
        "number of labels in imbalanced setting",
    )
    parser.add_argument(
        "--uratio",
        type=int,
        default=1,
        help="the ratio of unlabeled data to labeled data in each mini-batch",
    )
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0)
    parser.add_argument("-nl", "--num_labels", type=int, default=3004)
    parser.add_argument(
        "-alg", "--algorithm", type=str, default="fixmatch", help="ssl algorithm"
    )
    parser.add_argument("-nc", "--num_classes", type=int, default=751)

    main()
