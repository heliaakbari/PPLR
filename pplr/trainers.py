from __future__ import print_function, absolute_import
import time
import torch
import copy
from .evaluation_metrics import accuracy
from semilearn.core.criterions import CELoss, ConsistencyLoss
from .loss import AALS, PGLR, SoftTripletLoss, CrossEntropyLabelSmooth
import torchvision.transforms as transforms
from semilearn.datasets.augmentation.randaugment import RandAugment
from semilearn.algorithms.hooks.masking import FixedThresholdingHook
from .utils.meters import AverageMeter


class PPLRTrainer(object):
    def __init__(self, model, score, num_class=500, num_part=6, beta=0.5, aals_epoch=5):
        super(PPLRTrainer, self).__init__()
        self.model = model
        self.score = score
        self.masking_hook = FixedThresholdingHook()
        self.num_class = num_class
        self.num_part = num_part
        self.aals_epoch = aals_epoch

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((384, 128), padding=(int(384 * (1 - 0.875)), int(128 * (1 - 0.875))), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 128)),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.RandomCrop((384, 128), padding=(int(384 * (1 - 0.875)), int(128 * (1 - 0.875))), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.criterion_pglr = PGLR().cuda()
        self.criterion_aals = AALS().cuda()
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()
        self.criterion_tri = SoftTripletLoss().cuda()
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()
        

    def train(self, epoch, lb_train_dataloader, ulb_train_dataloader, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        losses_gce = AverageMeter()
        losses_tri = AverageMeter()
        losses_pce = AverageMeter()
        losses_fix = AverageMeter()
        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):

            data_ulb = ulb_train_dataloader.next()
            data_lb = lb_train_dataloader.next()
            #print(f"data_ulb in {i} iteration: {data_ulb[2]}")
            #print(f"data_lb in {i} iteration: {data_lb[2]}")
            ulb_x, ulb_targets, ulb_ca = self._parse_data(data_ulb)
            lb_x, lb_targets, lb_ca = self._parse_data(data_lb)
            

            # feedforward
            lb_emb_g, lb_emb_p, lb_logits_g, lb_logits_p = self.model(lb_x)
            lb_logits_g, lb_logits_p = lb_logits_g[:, :self.num_class], lb_logits_p[:, :self.num_class, :]

            ulb_emb_g, ulb_emb_p, ulb_logits_g, ulb_logits_p = self.model(ulb_x)
            ulb_logits_g, ulb_logits_p = ulb_logits_g[:, :self.num_class], ulb_logits_p[:, :self.num_class, :]

            # loss
            lb_loss_gce = self.criterion_pglr(lb_logits_g, lb_logits_p, lb_targets, lb_ca, lam=1)
            ulb_loss_gce = self.criterion_pglr(ulb_logits_g, ulb_logits_p, ulb_targets, ulb_ca, lam=0.5)
            loss_gce = lb_loss_gce + ulb_loss_gce

            emb_g = torch.cat([lb_emb_g, ulb_emb_g], dim=0)
            targets = torch.cat([lb_targets, ulb_targets], dim=0)
            loss_tri = self.criterion_tri(emb_g, targets)

            # fixmatch loss
            """
            mask = self.masking_hook.masking('fixmatch', logits_x_ulb=ulb_targets, softmax_x_ulb=False)

            self.model.eval()
            lb_emb_g_w, lb_logits_g_w = self.model(lb_x_w)
            ulb_emb_g_s, ulb_logits_g_s = self.model(ulb_x_s)

            self.model.train()

            sup_loss = self.ce_loss(lb_logits_g_w, lb_targets, reduction='mean')
            unsup_loss = self.consistency_loss(ulb_logits_g_s,
                                               ulb_targets,
                                               'ce',
                                               mask=mask)
            fixmatch_loss = sup_loss + unsup_loss
            """

            loss_pce = 0.
            lb_loss_pce = 0.
            ulb_loss_pce = 0.
            logits_p = torch.cat([lb_logits_p, ulb_logits_p], dim=0)
            ca = torch.cat([lb_ca, ulb_ca], dim=0)
            if self.num_part > 0:
                if epoch >= self.aals_epoch:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_aals(logits_p[:, :, part], targets, ca[:, part])
                else:
                    for part in range(self.num_part):
                        lb_loss_pce += self.criterion_ce(lb_logits_p[:, :, part], lb_targets, epsilon=1)
                        ulb_loss_pce += self.criterion_ce(ulb_logits_p[:, :, part], ulb_targets, epsilon=0.1)
                        loss_pce += lb_loss_pce + ulb_loss_pce
                loss_pce /= self.num_part


            loss = loss_gce + loss_tri + loss_pce #  + fixmatch_loss

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summing-up
            logits_g = torch.cat([lb_logits_g, ulb_logits_g], dim=0)
            prec, = accuracy(logits_g.data, targets.data)

            losses_gce.update(loss_gce.item())
            losses_tri.update(loss_tri.item())
            losses_pce.update(loss_pce.item())
            #losses_fix.update(fixmatch_loss.item())
            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                #print(f"data_ulb pid in {i} iteration: {data_ulb[2]}")
                #print(f"data_lb pid in {i} iteration: {data_lb[2]}")
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'L_GCE {:.3f} ({:.3f})\t'
                      'L_PCE {:.3f} ({:.3f})\t'
                      'L_TRI {:.3f} ({:.3f})\t'
                      'L_FIX {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(lb_train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_gce.val, losses_gce.avg,
                              losses_pce.val, losses_pce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_fix.val, losses_fix.avg,
                              precisions.val, precisions.avg))
                print('    └─> Breakdown: LB_GCE: {:.3f}, ULB_GCE: {:.3f} | LB_PCE: {:.3f}, ULB_PCE: {:.3f}'
                      .format(lb_loss_gce.item(), 
                              ulb_loss_gce.item(), 
                              lb_loss_pce.item() if isinstance(lb_loss_pce, torch.Tensor) else lb_loss_pce,
                              ulb_loss_pce.item() if isinstance(ulb_loss_pce, torch.Tensor) else ulb_loss_pce))

    def _parse_data(self, inputs):
        imgs, _, pids, _, idxs, is_lb = inputs
        if is_lb[0].item():
            ca = torch.ones((is_lb.shape[0], 3), dtype=torch.float32)
            #w_imgs = torch.stack([self.transform(img) for img in imgs])
            return imgs.cuda(), pids.cuda(), ca.cuda()
        else:
            #s_imgs = torch.stack([self.strong_transform(img) for img in imgs])
            ca = self.score[idxs]
            return imgs.cuda(), pids.cuda(), ca.cuda()
        


class PPLRTrainerCAM(object):
    def __init__(self, model, score, memory, memory_p, num_class=500, num_part=6, beta=0.5, aals_epoch=5, lam_cam=0.5):
        super(PPLRTrainerCAM, self).__init__()
        self.model = model
        self.score = score
        self.memory = memory
        self.memory_p = memory_p

        self.num_class = num_class
        self.num_part = num_part
        self.lam_cam = lam_cam
        self.aals_epoch = aals_epoch

        self.criterion_pglr = PGLR(lam=beta).cuda()
        self.criterion_aals = AALS().cuda()
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()
        self.criterion_tri = SoftTripletLoss().cuda()

    def train(self, epoch, train_dataloader, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        losses_gce = AverageMeter()
        losses_tri = AverageMeter()
        losses_cam = AverageMeter()
        losses_pce = AverageMeter()

        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):
            data = train_dataloader.next()
            inputs, targets, cams, ca = self._parse_data(data)

            # feedforward
            emb_g, emb_p, logits_g, logits_p = self.model(inputs)
            logits_g, logits_p = logits_g[:, :self.num_class], logits_p[:, :self.num_class, :]

            # loss
            loss_gce = self.criterion_pglr(logits_g, logits_p, targets, ca)
            loss_tri = self.criterion_tri(emb_g, targets)
            loss_gcam = self.memory(emb_g, targets, cams)

            loss_pce = 0.
            loss_pcam = 0.
            if self.num_part > 0:
                if epoch >= self.aals_epoch:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_aals(logits_p[:, :, part], targets, ca[:, part])
                        loss_pcam += self.memory_p[part](emb_p[:, :, part], targets, cams)
                else:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_ce(logits_p[:, :, part], targets)
                        loss_pcam += self.memory_p[part](emb_p[:, :, part], targets, cams)
                loss_pce /= self.num_part
                loss_pcam /= self.num_part

            loss_cam = loss_pcam + loss_gcam
            loss = loss_gce + loss_pce + loss_tri + loss_cam * self.lam_cam

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summing-up
            prec, = accuracy(logits_g.data, targets.data)

            losses_gce.update(loss_gce.item())
            losses_tri.update(loss_tri.item())
            losses_cam.update(loss_cam.item())
            losses_pce.update(loss_pce.item())
            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'L_GCE {:.3f} ({:.3f})\t'
                      'L_PCE {:.3f} ({:.3f})\t'
                      'L_TRI {:.3f} ({:.3f})\t'
                      'L_CAM {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_gce.val, losses_gce.avg,
                              losses_pce.val, losses_pce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_cam.val, losses_cam.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, cids, idxs = inputs
        ca = self.score[idxs]
        return imgs.cuda(), pids.cuda(), cids.cuda(), ca.cuda()
