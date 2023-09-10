# Copyright (c) 2023 Chuanze Lu
import logging

import os
import re
import yaml
import torch


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    # if torch.cuda.is_available():
    #     logging.info('checkpoint loading from %s for gpu' % path)
    #     checkpoint = torch.load(path)
    # else:
    #     logging.info('checkpoint loading from %s for cpu' % path)
    #     checkpoint = torch.load(path, map_location='cpu')

    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint)


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    logging.info('checkpoint savc to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)






