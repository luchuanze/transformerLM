# Copyright (c) 2023 Chuanze Lu
import argparse
import os

import torch.jit
import yaml
from net.model import create_model
from net.checkpoint import load_checkpoint
from net.dataset.dataset import read_symbol_table

def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--dict_file', required=True, help='dict file')
    parser.add_argument('--checkpoint', required=True, help='model checkpoint')
    parser.add_argument('--output_file', required=True, help='output_file')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    symbol_table = read_symbol_table(args.dict_file)
    vocab_size = len(symbol_table)
    model_conf = configs.get('model_conf', {})
    model_conf['vocab_size'] = vocab_size
    model = create_model(model_conf)
    print(model)

    load_checkpoint(model, args.checkpoint)

    script_model = torch.jit.script(model)
    script_model.save(args.output_file)

    print('Export script model {}'.format(args.output_file))


if __name__ == '__main__':
    main()



