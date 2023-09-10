
import argparse
import os

import torch
from dataset.dataset import TextDataset, read_symbol_table
from model import create_model, add_sos_eos
from checkpoint import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='Do Training...')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pinned memory buffers used for reading')
    parser.add_argument('--num_workers', type=int, default=0, help='num of subprocess workers for processing data')
    parser.add_argument('--symbol_table', required=True, help='model unit symbol')
    parser.add_argument('--checkpoint', default=None, help='checkpoint model')

    args = parser.parse_args()
    return args


def main():

    args = get_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = S

