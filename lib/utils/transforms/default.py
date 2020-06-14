#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch
import torchvision.transforms as transforms

from ..loggers import STDLogger as logger
from ...core.config import Config as cfg

class DefaultTransform(transforms.Compose):

    @staticmethod
    def require_args():

        cfg.add_argument('--tfm-size', default="224", type=eval,
                            help='Training image size')
        cfg.add_argument('--tfm-resize', default="256", type=eval,
                            help='resize into')
        cfg.add_argument('--tfm-scale', default=None, type=eval,
                            help='scale for random resize crop')
        cfg.add_argument('--tfm-ratio', default="(3./4., 4./3.)", type=eval,
                            help='ratio for random resize crop')
        cfg.add_argument('--tfm-colorjitter', default=None, type=eval,
                            help='color jitters for input')
        cfg.add_argument('--tfm-random-grayscale', default=0, type=float,
                            help='transform input to gray scale')
        cfg.add_argument('--tfm-random-hflip', default=0, type=float,
                            help='random horizontally flip for input')
        cfg.add_argument('--tfm-means', default=None, type=eval,
                            help='channel-wise means for input sampels')
        cfg.add_argument('--tfm-stds', default=None, type=eval,
                            help='channel-wise stds for input sampels')

    def __init__(self, train, means=None, stds=None,
            size=None, resize=None, scale=None, ratio=None, 
            colorjitter=None, random_grayscale=None, random_hflip=None, tencrops=False):
        means = means if means is not None else cfg.tfm_means
        stds = stds if stds is not None else cfg.tfm_stds
        size = size if size is not None else cfg.tfm_size
        resize = resize if resize is not None else cfg.tfm_resize
        scale = scale if scale is not None else cfg.tfm_scale
        ratio = ratio if ratio is not None else cfg.tfm_ratio
        colorjitter = colorjitter if colorjitter is not None else cfg.tfm_colorjitter
        random_grayscale = random_grayscale if random_grayscale is not None else cfg.tfm_random_grayscale
        random_hflip = random_hflip if random_hflip is not None else cfg.tfm_random_hflip

        self.transforms = []
        if train:
            # size transform
            if scale is not None:
                logger.debug('Training samples will be '
                    'random resized with scale [%s] and ratio [%s] '
                    'then cropped to size [%s]' 
                    % (scale, ratio, size))
                self.transforms.append(transforms.RandomResizedCrop(size=size, 
                    scale=scale, ratio=ratio))
            else:
                logger.debug('Training samples will be resized to [%s] and then '
                    'random cropped into [%s]' % (resize, size))
                self.transforms.append(transforms.Resize(size=resize))
                self.transforms.append(transforms.RandomCrop(size))
            # colorjitter
            if colorjitter is not None:
                logger.debug('Training samples will be enhanced with colorjitter: '
                    '[%s]' % str(colorjitter))
                self.transforms.append(transforms.ColorJitter(*colorjitter))
            # grayscale
            if random_grayscale > 0:
                logger.debug('Training samples will be randomly converted to '
                    'grayscale with probability [%f]' % random_grayscale)
                self.transforms.append(transforms.RandomGrayscale(p=random_grayscale))
            # random hflip
            if random_hflip > 0:
                logger.debug('Training samples will be random horizontally flip '
                    'with probability [%f]' % random_hflip)
                self.transforms.append(transforms.RandomHorizontalFlip(p=random_hflip))
        else:
            self.transforms.append(transforms.Resize(resize))
            if not tencrops:
                logger.debug('Testing samples will be resized to [%s] and then '
                    'center crop to [%s]' % (resize, size))
                self.transforms.append(transforms.CenterCrop(size))
            else:
                self.transforms.append(transforms.TenCrop(size))
                logger.debug('Testing sampels will be resized to [%s] and then '
                    'ten cropped to [%s]' % (resize, size))


        to_tensor = transforms.ToTensor()
        # to tensor and normalize
        if means is not None and stds is not None:
            logger.debug('Samples will be normalised with means: [%s] '
                'and stds: [%s]' % (means, stds))
            normalise = transforms.Normalize(means, stds)
            if train or not tencrops:
                self.transforms.append(to_tensor)
                self.transforms.append(normalise)
            else:
                self.transforms.append(transforms.Lambda(lambda crops:torch.stack(
                    [normalise(to_tensor(crop)) for crop in crops])))
        else:
            logger.debug('Samples will not be normalised')
            if train or not tencrops:
                self.transforms.append(to_tensor)
            else:
                self.transforms.append(transforms.Lambda(lambda crops:torch.stack(
                    [to_tensor(crop) for crop in crops])))
