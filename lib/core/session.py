#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import os
import time
import random
import numpy as np
import importlib
from easydict import EasyDict as ezdict

import torch

from .config import Config as cfg
from ..utils.loggers import STDLogger as logger

class Session:

    def __init__(self, main):
        # require session arguments
        self.require_args()
        # load main module
        self.main_module = importlib.import_module(main)
        # parse args
        if hasattr(self.main_module, 'require_args'):
            self.main_module.require_args()
        cfg.parse()

    def run(self):

        # setup session according to args
        self.setup()

        # run main function
        self.main_module.main()

    def require_args(self):

        # timestamp
        stt = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
        tt = int(time.time())

        cfg.add_argument('--session', default=stt, type=str,
                            help='session name')
        cfg.add_argument('--session-root', default='./sessions', type=str,
                            help='root directory to store session')
        cfg.add_argument('--seed', default=None, type=int,
                            help='random seed')
        cfg.add_argument('--brief', action='store_true',
                            help='print log with priority higher than debug')
        cfg.add_argument('--debug', action='store_true',
                            help='no generated files will be stored')
        cfg.add_argument('--gpus', default='', type=str,
                            help='available gpu list, leave empty to use cpu')
        cfg.add_argument('--resume', default=None, type=str,
                            help='path to resume session')
        cfg.add_argument('--restart', action='store_true',
                            help='load session and start a new one')

    def setup(self):
        """
        set up common environment for training
        """

        # show session name at first
        logger.debug('Current session name is [%s]' % cfg.session)

        # check running mode
        if cfg.debug:
            logger.warn('Session will be ran in debug mode, '
                        'no generated files should be stored including '
                        'checkpoint and log')

        # print args
        if not cfg.brief:
            logger.debug('Session will be ran with following arguments:\n%s' % cfg)
        else:
            logger.debug('Session will be ran in [BRIEF] mode '
                         'in which most of debugging messages will not be printed')

        # if not verbose, set log level to info
        logger.setup(logger.INFO if cfg.brief else logger.DEBUG)

        # fix random seeds
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
            torch.backends.cudnn.deterministic = True
            logger.debug('Random seed will be fixed to [%d]' % cfg.seed)
        else:
            logger.debug('Random seed will not be fixed')
        
        # set visible gpu devices at main function
        cfg.gpus = cfg.gpus.strip()
        if len(cfg.gpus) > 0:
            logger.debug('Visible gpu devices are: [%s]' % cfg.gpus)
        else:
            logger.warn('No available gpu found, session will be ran on [CPU]')
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # setup session name and session path
        if cfg.resume and cfg.resume.strip() != '' and not cfg.restart:
            assert os.path.exists(cfg.resume), ("Resume file not "
                                                    "found: %s" % cfg.resume)
            ckpt = torch.load(cfg.resume)
            if 'session' in ckpt:
                cfg.session = ckpt['session']
                logger.debug('Session name changes to [%s]' % cfg.session)
        cfg.session_dir = os.path.join(cfg.session_root, cfg.session)

        # setup checkpoint dir
        cfg.ckpt_dir = os.path.join(cfg.session_dir, 'checkpoint')
        if not os.path.exists(cfg.ckpt_dir) and not cfg.debug:
            logger.debug('Checkpoint files will be stored in %s' % cfg.ckpt_dir)
            os.makedirs(cfg.ckpt_dir)

        # redirect logs to file
        cfg.log_dir = os.path.join(cfg.session_dir, 'log.txt')
        if not cfg.debug:
            logger.debug('Log files will be stored in %s' % cfg.log_dir)
            logger.setup(to_file=cfg.log_dir)

        # setup tfb log dir
        cfg.tfb_dir = os.path.join(cfg.session_dir, 'tfboard')
        if not os.path.exists(cfg.tfb_dir) and not cfg.debug:
            logger.debug('TFboard files will be stored in %s if applicable' % cfg.tfb_dir)
            os.makedirs(cfg.tfb_dir)

        # store options at log directory
        if not cfg.debug:
            cfg_path = os.path.join(cfg.session_dir, 'config.yaml')
            logger.debug('Provided arguments will be stored in %s' % cfg_path)
            with open(cfg_path, 'w') as out:
                out.write(cfg.yaml())