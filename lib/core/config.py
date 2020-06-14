#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import os
import argparse
import yaml
import importlib
import traceback
from prettytable import PrettyTable
from easydict import EasyDict as ezdict

class _META_(type):

    # real argparser instance
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parsed arguments
    ARGS = ezdict()
    # monitored modules
    MODULES = ezdict()

    def require_args(self):

        # args for config file
        _META_.PARSER.add_argument('--cfgs', type=str, nargs='*',
                            help='config files to load')

    def register_module(self, name, path):
        _META_.MODULES[name] = ezdict(path=path, classes=ezdict())

    def register_class(self, module, name, obj):
        assert module in _META_.MODULES.keys(), ('No module '
            'named [%s] has been registered' % module)
        _META_.MODULES[module].classes[name] = obj

    def get_class_name(self, module):
        assert module in _META_.MODULES.keys(), ('No module '
            'named [%s] has been registered' % module)
        return [name for name,_ in _META_.MODULES[module].classes.iteritems()]

    def get_class(self, module, name=None):
        assert module in _META_.MODULES.keys(), ('No module '
            'named [%s] has been registered' % module)
        if name is None:
            return [obj for name,obj in _META_.MODULES[module].classes.iteritems()]
        assert name in _META_.MODULES[module].classes.keys(), ('No class named [%s] '
            'has been registered in module [%s]' % (name, module))
        return _META_.MODULES[module].classes[name]

    def parse(self):
        # collect self args
        self.require_args()

        # load default args from config file
        self.from_files(self.known_args().cfgs)

        # collect args for packages
        for module in _META_.MODULES.keys():
            # setup module arguments
            mod = importlib.import_module(_META_.MODULES[module].path)
            if hasattr(mod, 'require_args'):
                mod.require_args()

        # re-update default value for new args
        self.from_files(self.known_args().cfgs)
                
        for module in _META_.MODULES.keys():
            # setup class arguments
            if hasattr(self.known_args(), module):
                cls = self.get_class(module, self.known_args().__dict__[module])
                if hasattr(cls, 'require_args'):
                    cls.require_args()
            else:
                cls_list = self.get_class(module)
                for cls in cls_list:
                    if hasattr(cls, 'require_args'):
                        cls.require_args()

        # re-update default value for new args
        self.from_files(self.known_args().cfgs)

        # parse args
        _META_.ARGS = _META_.PARSER.parse_args()

    def known_args(self):
        args, _ = _META_.PARSER.parse_known_args()
        return args

    def from_files(self, files):

        # if no config file is provided, skip
        if files is None or len(files) <= 0:
            return None

        for file in files:
            assert os.path.exists(file), "Config file not found: [%s]" % file
            configs = yaml.load(open(file, 'r'))
            _META_.PARSER.set_defaults(**configs)

    def get(self, attr, default=None):
        if hasattr(_META_.ARGS, attr):
            return getattr(_META_.ARGS, attr)
        return default

    def set(self, key, val):
        setattr(_META_.ARGS, key, val)

    def yaml(self):
        config = {k:v for k,v in sorted(vars(_META_.ARGS).items())}
        return yaml.safe_dump(config, default_flow_style=False)

    def __getattr__(self, attr):
        try:
            return _META_.PARSER.__getattribute__(attr)
        except AttributeError:
            return _META_.ARGS.__getattribute__(attr)
        except:
            traceback.print_exec()
            exit(-1)

    def __str__(self):
        MAX_WIDTH = 20
        table = PrettyTable(["#", "Key", "Value", "Default"])
        table.align = 'l'
        for i, (k, v) in enumerate(sorted(vars(_META_.ARGS).items())):
            v = str(v)
            default = str(_META_.PARSER.get_default(k))
            if default == v:
                default = '--'
            table.add_row([i, k, v[:MAX_WIDTH] + ('...' if len(v) > MAX_WIDTH else ''), default])
        return table.get_string()

class Config(object):
    __metaclass__ = _META_