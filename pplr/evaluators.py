from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import random
import copy


import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs[0].data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def extract_all_features(model, data_loader, print_freq=200):
    correctLabels= 0
    print("enter extract all features")
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features_g = OrderedDict()
    features_p = OrderedDict()
    labels = OrderedDict()
    classes = OrderedDict()

    end = time.time()
    train_transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((384, 128), padding=(int(384 * (1 - 0.875)), int(128 * (1 - 0.875))), padding_mode='reflect'),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _, is_lb) in enumerate(data_loader):
            #print(f"pids in extract all features:  {pids}")
            data_time.update(time.time() - end)
            
            #w_imgs = torch.stack([train_transform(img) for img in imgs])
            inputs = to_torch(imgs).cuda()
            #w_inputs = to_torch(w_imgs).cuda()

            #print("1",inputs.shape) #torch.Size([5, 3, 384, 128])
            if isinstance(model, nn.DataParallel):
                outputs_g, outputs_p = model.module.extract_all_features(inputs)
                
                logits = model.module.extract_global_classes(inputs)
            else:
                outputs_g, outputs_p = model.extract_all_features(inputs)
                
                logits = model.extract_global_classes(inputs)
            #print("2",inputs.shape) #torch.Size([5, 3, 384, 128])
            outputs_g, outputs_p, y_logits = outputs_g.data.cpu(), outputs_p.data.cpu(), logits.data.cpu()
            
            for fname, output_g, output_p, y_logit, pid, in zip(fnames, outputs_g, outputs_p, y_logits, pids):
                features_g[fname] = output_g
                features_p[fname] = output_p
                labels[fname] = pid
                classes[fname] = torch.max(y_logit, dim=-1)[1].cpu().tolist()
                #print(input_.shape) #torch.Size([3, 384, 128])
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
        #print(features_g.shape, features_p.shape, images.shape, labels.shape)
        print(f"number of correct labels: {correctLabels}")
        return features_g, features_p, classes, labels

def return_batch_images(model, data_loader, print_freq=200):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features_g = OrderedDict()
    features_p = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs = to_torch(imgs).cuda()

        return inputs


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def visualize_ranked_results(distmat, query, gallery, save_dir, topk=3):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    distmat = distmat.numpy() if isinstance(distmat, torch.Tensor) else distmat

    query_fnames = [f for f, _, _ in query]
    gallery_fnames = [f for f, _, _ in gallery]
    gallery_ids = [pid for _, pid, _ in gallery]

    for i, q_fname in enumerate(query_fnames):
        q_pid = query[i][1]
        q_img = Image.open(q_fname).convert('RGB')
        
        sorted_idxs = np.argsort(distmat[i])
        topk_idxs = [idx for idx in sorted_idxs if gallery_ids[idx] != q_pid][:topk]

        fig, axes = plt.subplots(1, topk + 1, figsize=(4*(topk + 1), 6))
        axes[0].imshow(q_img)
        axes[0].set_title(f"Query\nID: {q_pid}", fontsize=10)
        axes[0].axis('off')

        for rank, idx in enumerate(topk_idxs):
            g_fname = gallery_fnames[idx]
            g_pid = gallery[idx][1]
            score = distmat[i][idx]
            g_img = Image.open(g_fname).convert('RGB')

            axes[rank + 1].imshow(g_img)
            axes[rank + 1].set_title(f"Rank-{rank+1}\nID: {g_pid}\n Distance: {score:.2f}", fontsize=10)
            axes[rank + 1].axis('off')

        save_path = os.path.join(save_dir, f"query_{i}_id_{q_pid}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, epoch, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        #if epoch % 10 == 0:
            #visualize_ranked_results(distmat, query, gallery,  save_dir=f'visual_results/epoch{epoch}/', topk=3)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
