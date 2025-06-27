from __future__ import print_function, absolute_import
import time
import torch
from .evaluation_metrics import accuracy
from semilearn.core.criterions import CELoss, ConsistencyLoss
from .loss import AALS, PGLR, SoftTripletLoss, CrossEntropyLabelSmooth
from .utils.meters import AverageMeter


class PPLRTrainer(object):
    def __init__(self, model, score, num_class=500, num_part=6, beta=0.5, aals_epoch=5):
        super(PPLRTrainer, self).__init__()
        self.model = model
        self.score = score

        self.num_class = num_class
        self.num_part = num_part
        self.aals_epoch = aals_epoch

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
        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):

            data_ulb = ulb_train_dataloader.next()
            data_lb = lb_train_dataloader.next()
            ulb_inputs, ulb_targets, ulb_ca = self._parse_data(data_ulb)
            lb_inputs, lb_targets, lb_ca = self._parse_data(data_lb)
            

            # feedforward
            lb_emb_g, lb_emb_p, lb_logits_g, lb_logits_p = self.model(lb_inputs)
            lb_logits_g, lb_logits_p = lb_logits_g[:, :self.num_class], lb_logits_p[:, :self.num_class, :]

            ulb_emb_g, ulb_emb_p, ulb_logits_g, ulb_logits_p = self.model(ulb_inputs)
            ulb_logits_g, ulb_logits_p = ulb_logits_g[:, :self.num_class], ulb_logits_p[:, :self.num_class, :]

            # loss
            lb_loss_gce = self.criterion_pglr(lb_logits_g, lb_logits_p, lb_targets, lb_ca, lam=1)
            ulb_loss_gce = self.criterion_pglr(ulb_logits_g, ulb_logits_p, ulb_targets, ulb_ca, lam=0.5)
            loss_gce = lb_loss_gce + ulb_loss_gce

            emb_g = lb_emb_g + ulb_emb_g
            targets = lb_targets + ulb_targets
            loss_tri = self.criterion_tri(emb_g, targets)

            # fixmatch loss
            #sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            #unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                           #pseudo_label,
                                           #'ce',
                                           #mask=mask)

            loss_pce = 0.
            lb_loss_pce = 0.
            ulb_loss_pce = 0.
            logits_p = lb_logits_p + ulb_logits_p
            ca = lb_ca + ulb_ca
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


            loss = loss_gce + loss_tri + loss_pce

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summing-up
            logits_g = lb_logits_g + ulb_logits_g
            prec, = accuracy(logits_g.data, targets.data)

            losses_gce.update(loss_gce.item())
            losses_tri.update(loss_tri.item())
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
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(lb_train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_gce.val, losses_gce.avg,
                              losses_pce.val, losses_pce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, idxs, is_lb = inputs
        if is_lb[0].item():
            ca = torch.ones((16, 3), dtype=torch.float32)
        else:
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
