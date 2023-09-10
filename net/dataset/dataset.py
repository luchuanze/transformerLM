# Copyright (c) 2023 Chuanze Lu
import codecs
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def read_symbol_table(unit_file):
    symbol_table = {}
    with open(unit_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


class TextData(Dataset):
    def __init__(self,
                 symbol_table,
                 data_file,
                 batch_conf, sort):

        #assert batch_type in ['static', 'dynamic']
        data = []
        with codecs.open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                # line_json = line.strip()
                # obj = json.loads(line_json)
                #
                # assert 'key' in obj
                # assert 'txt' in obj
                #
                # key = obj['key']
                # txt = obj['txt']
                line_arr = line.strip().split()

                key = line_arr[0]
                txt = line_arr[1]

                tokens = []
                valid = True
                for ch in txt:
                    if ch == ' ':
                        ch = '_'
                    if ch in symbol_table:
                        tokens.append(symbol_table[ch])
                    elif '<unk>' in symbol_table:
                        valid = False
                        tokens.append(symbol_table['<unk>'])

                if valid is True:
                    data.append((key, tokens))

        # if sort:
            # data = sorted(data, key=lambda x: x[3])
        # random.shuffle(data)
        data = sorted(data, key=lambda x: len(x[1]))

        self.minibatch = []
        num = len(data)
        assert 'batch_size' in batch_conf
        batch_size = batch_conf['batch_size']

        cur = 0
        while cur < num:
            end = min(cur + batch_size, num)
            item = []

            for i in range(cur, end):
                item.append(data[i])

            self.minibatch.append(item)
            cur = end

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, item):
        return self.minibatch[item]


class CollateFunc(object):
    def __init__(self,
                 shuffle,
                 shuffle_conf,
                 sort,
                 sort_conf
                 ):

        self.shuffle = shuffle
        self.shuffle_conf = shuffle_conf
        self.sort = sort
        self.sort_conf = sort_conf

    def __call__(self, batch):
        assert len(batch) == 1

        keys = []
        tokens = []
        batch_item = batch[0]
        for i, x in enumerate(batch_item):
            keys.append(x[0])
            tokens.append(x[1])

        # ys = tokens

        ys = [np.array(tokens[i], dtype=np.int32) for i in range(0, len(batch_item))]
        #padding
        if ys is None:
            ys_pad = None
            ys_lengths = None
        else:
            ys_lengths = torch.from_numpy(
                np.array([y.shape[0] for y in ys], dtype=np.int32))
            if len(ys) > 0:
                ys_pad = pad_sequence([torch.from_numpy(y).long() for y in ys], True, -1)

        return keys, ys_pad, ys_lengths


class TextDataset(object):
    def __init__(self,
                 distributed,
                 symbol_table,
                 data_file,
                 batch_conf,
                 filter_conf,
                 shuffle,
                 shuffle_conf,
                 sort,
                 sort_conf
                 ):
        self.distributed = distributed
        self.dataset = TextData(symbol_table, data_file, batch_conf, sort)
        self.collate = CollateFunc(
                 shuffle,
                 shuffle_conf,
                 sort,
                 sort_conf)
        if distributed:
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, shuffle=True
            )
        else:
            self.sampler = None

    def get_loader(self,
                  pin_memory,
                  num_workers):

        return DataLoader(self.dataset,
                          collate_fn=self.collate,
                          sampler=self.sampler,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          batch_size=1)

    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)





