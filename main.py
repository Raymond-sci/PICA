#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import os
import sys
sys.path.append('..')
import time
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from lib import Config as cfg, networks, datasets, Session
from lib.utils import (lr_policy, optimizers, transforms, save_checkpoint, 
                            AverageMeter, TimeProgressMeter, traverse)
from lib.utils.loggers import STDLogger as logger, TFBLogger as SummaryWriter

from pica.utils import ConcatDataset, RepeatSampler, RandomSampler, get_reduced_transform
from pica.losses import PUILoss

def require_args():

    # args for training
    cfg.add_argument('--max-epochs', default=200, type=int,
                        help='maximal training epoch')
    cfg.add_argument('--display-freq', default=80, type=int,
                        help='log display frequency')
    cfg.add_argument('--batch-size', default=256, type=int,
                        help='size of mini-batch')
    cfg.add_argument('--num-workers', default=4, type=int,
                        help='number of workers used for loading data')
    cfg.add_argument('--data-nrepeat', default=1, type=int,
                        help='how many times each image in a ' +
                             'mini-batch should be repeated')
    cfg.add_argument('--pica-lamda', default=2.0, type=float,
                        help='weight of negative entropy regularisation')

def main():

    logger.info('Start to declare training variable')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Session will be ran in device: [%s]' % cfg.device)
    start_epoch = 0
    best_acc = 0.

    logger.info('Start to prepare data')
    # get transformers
    # train_transform is for data perturbation
    train_transform = transforms.get(train=True)
    # test_transform is for evaluation
    test_transform = transforms.get(train=False)
    # reduced_transform is for original training data
    reduced_transform = get_reduced_transform(cfg.tfm_resize, cfg.tfm_size, 
                                                cfg.tfm_means, cfg.tfm_stds)
    # get datasets
    # each head should have its own trainset
    train_splits = dict(cifar100=[['train', 'test']],
        stl10=[['train+unlabeled', 'test'], ['train', 'test']])
    test_splits = dict(cifar100=['train', 'test'],
        stl10=['train', 'test'])
    # instance dataset for each head
    # otrainset: original trainset
    otrainset = [ConcatDataset([datasets.get(split=split, transform=reduced_transform) 
                    for split in train_splits[cfg.dataset][hidx]]) 
                    for hidx in xrange(len(train_splits[cfg.dataset]))]
    # ptrainset: perturbed trainset
    ptrainset = [ConcatDataset([datasets.get(split=split, transform=train_transform) 
                    for split in train_splits[cfg.dataset][hidx]]) 
                    for hidx in xrange(len(train_splits[cfg.dataset]))]
    # testset
    testset = ConcatDataset([datasets.get(split=split, transform=test_transform) 
                    for split in test_splits[cfg.dataset]])
    # declare data loaders for testset only
    test_loader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, 
                                num_workers=cfg.num_workers)

    logger.info('Start to build model')
    net = networks.get()
    criterion = PUILoss(cfg.pica_lamda)
    optimizer = optimizers.get(params=[val for _, val in net.trainable_parameters().iteritems()])
    lr_handler = lr_policy.get()

    # load session if checkpoint is provided
    if cfg.resume:
        assert os.path.exists(cfg.resume), "Resume file not found"
        ckpt = torch.load(cfg.resume)
        logger.info('Start to resume session for file: [%s]' % cfg.resume)
        net.load_state_dict(ckpt['net'])
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

    # data parallel
    if cfg.device == 'cuda' and len(cfg.gpus.split(',')) > 1:
        logger.info('Data parallel will be used for acceleration purpose')
        device_ids = range(len(cfg.gpus.split(',')))
        if not (hasattr(net, 'data_parallel') and net.data_parallel(device_ids)):
            net = nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True
    else:
        logger.info('Data parallel will not be used for acceleration')

    # move modules to target device
    net, criterion = net.to(cfg.device), criterion.to(cfg.device)

    # tensorboard wrtier
    writer = SummaryWriter(cfg.debug, log_dir=cfg.tfb_dir)
    # start training
    lr = cfg.base_lr
    epoch = start_epoch
    while lr > 0 and epoch < cfg.max_epochs:

        lr = lr_handler.update(epoch, optimizer)
        writer.add_scalar('Train/Learing_Rate', lr, epoch)

        logger.info('Start to train at %d epoch with learning rate %.5f' % (epoch, lr))
        train(epoch, net, otrainset, ptrainset, optimizer, criterion, writer)

        logger.info('Start to evaluate after %d epoch of training' % epoch)
        acc, nmi, ari = evaluate(net, test_loader)
        logger.info('Evaluation results at epoch %d are: '
            'ACC: %.3f, NMI: %.3f, ARI: %.3f' % (epoch, acc, nmi, ari))
        writer.add_scalar('Evaluate/ACC', acc, epoch)
        writer.add_scalar('Evaluate/NMI', nmi, epoch)
        writer.add_scalar('Evaluate/ARI', ari, epoch)

        epoch += 1

        if cfg.debug:
            continue

        # save checkpoint
        is_best = acc > best_acc
        best_acc = max(best_acc, acc)
        save_checkpoint({'net' : net.state_dict(), 
                'optimizer' : optimizer.state_dict(),
                'acc' : acc,
                'epoch' : epoch}, is_best=is_best)

    logger.info('Done')

def train(epoch, net, otrainset, ptrainset, optimizer, criterion, writer):
    """alternate the training of different heads
    """
    for hidx, head in enumerate(cfg.net_heads):
        train_head(epoch, net, hidx, head, otrainset[min(len(otrainset) - 1, hidx)], 
            ptrainset[min(len(ptrainset) - 1, hidx)], optimizer, criterion, writer)

def train_head(epoch, net, hidx, head, otrainset, ptrainset, optimizer, criterion, writer):
    """trains one head for an epoch
    """

    # declare dataloader
    random_sampler = RandomSampler(otrainset)
    batch_sampler = RepeatSampler(random_sampler, cfg.batch_size, nrepeat=cfg.data_nrepeat)
    ploader = DataLoader(ptrainset, batch_sampler=batch_sampler, 
                        num_workers=cfg.num_workers, pin_memory=True)
    oloader = DataLoader(otrainset, sampler=random_sampler, 
                        batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
                        pin_memory=True)
    
    # set network mode
    net.train()

    # tracking variable
    end = time.time()
    train_loss = AverageMeter('Loss', ':.4f')
    data_time = AverageMeter('Data', ':.3f')
    batch_time = AverageMeter('Time', ':.3f')
    progress = TimeProgressMeter(batch_time, data_time, train_loss, 
            Batch=len(oloader), Head=len(cfg.net_heads), Epoch=cfg.max_epochs)

    for batch_idx, (obatch, pbatch) in enumerate(itertools.izip(oloader, ploader)):
        # record data loading time
        data_time.update(time.time() - end)

        # move data to target device
        (oinputs, _), (pinputs, _) = (obatch, pbatch)
        oinputs, pinputs = (oinputs.to(cfg.device, non_blocking=True), 
                            pinputs.to(cfg.device, non_blocking=True))

        # forward
        ologits, plogits = net(oinputs)[hidx], net(pinputs)[hidx]
        loss = criterion(ologits.repeat(cfg.data_nrepeat, 1), plogits)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), oinputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('Train/Loss/Head-%d' % head, train_loss.val, epoch * len(oloader) + batch_idx)

        if batch_idx % cfg.display_freq != 0:
            continue

        logger.info(progress.show(Batch=batch_idx, Epoch=epoch, Head=hidx))

def evaluate(net, loader):
    """evaluates on provided data
    """

    net.eval()
    predicts = np.zeros(len(loader.dataset), dtype=np.int32)
    labels = np.zeros(len(loader.dataset), dtype=np.int32)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            logger.progress('processing %d/%d batch' % (batch_idx, len(loader)))
            inputs = inputs.to(cfg.device, non_blocking=True)
            # assuming the last head is the main one
            # output dimension of the last head 
            # should be consistent with the ground-truth
            logits = net(inputs)[-1]
            start = batch_idx * loader.batch_size
            end = start + loader.batch_size
            end = min(end, len(loader.dataset))
            labels[start:end] = targets.cpu().numpy()
            predicts[start:end] = logits.max(1)[1].cpu().numpy()

    # compute accuracy
    num_classes = labels.max().item() + 1
    count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in xrange(predicts.shape[0]):
        count_matrix[predicts[i], labels[i]] += 1
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]
    acc = count_matrix[reassignment[:,0], reassignment[:,1]].sum().astype(np.float32) / predicts.shape[0]
    return acc, NMI(labels, predicts), ARI(labels, predicts)


if __name__ == '__main__':
    Session(__name__).run()
