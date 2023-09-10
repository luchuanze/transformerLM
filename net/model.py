# Copyright (c) 2023 Chuanze Lu
import logging

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from net.transfomerlm import TransformerLM

from torch.nn.utils.rnn import pad_sequence
import numpy as np


class LmModel(TransformerLM):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 attention_heads,
                 num_layers,
                 dropout):

        super(LmModel, self).__init__(vocab_size,
                                      embed_size,
                                      attention_heads,
                                      num_layers,
                                      dropout)

    @torch.jit.export
    def inference_loss(self,
                       x,
                       x_len,
                       sos_id: int = -1,
                       eos_id: int = -1,
                       ignore_id: int = -1) -> torch.Tensor:
        src, dest = add_sos_eos(x, sos_id, eos_id, ignore_id)
        src_len = x_len + 1
        # with torch.no_grad():
        logits = self.forward(src, src_len)
        # #     # logits = torch.log_softmax(logits, dim=1)
        loss = criterion(logits, dest)

        return loss


def create_model(configs):
    vocab_size = configs['vocab_size']
    embed_size = configs['embed_size']
    hidden_size = configs['hidden_size']
    num_layers = configs['num_layers']
    attention_heads = configs['attention_heads']
    dropout = configs['dropout']

    lm_model = LmModel(vocab_size,
                             embed_size,
                             attention_heads,
                             num_layers,
                             dropout)

    return lm_model


def input_tokenizer(symbol_table, input_text, device):
    tokens = []
    for ch in input_text:
        if ch == ' ':
            ch = '_'
        if ch in symbol_table:
            tokens.append(symbol_table[ch])
        elif '<unk>' in symbol_table:
            print(ch)
            tokens.append(symbol_table['<unk>'])

    x = torch.from_numpy(np.array(tokens, dtype=np.int32)).long().to(device)
    x_len = torch.from_numpy(np.array([len(tokens)], dtype=np.int32)).to(device)
    x = pad_sequence([x], True, -1)

    return x, x_len


def cross_entropy(logits: torch.Tensor,
                  target: torch.Tensor,
                  ignore_index: int = -1) -> torch.Tensor:

    return nn.functional.cross_entropy(logits, target, ignore_index=ignore_index)


def criterion(logits: torch.Tensor,
              target: torch.Tensor,
              ignore_index: int = -1,
              smoothing: float = 0.1) -> torch.Tensor:
    class_size = logits.size(2)
    logits = logits.view(-1, class_size)
    target = target.view(-1)

    # true_dist = torch.zeros_like(logits)
    # true_dist.fill_(smoothing/(class_size - 1))
    # ignore = target == ignore_index
    # total = len(target) - ignore.sum().item()
    # target = target.masked_fill(ignore, 0)
    # true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)

    loss = cross_entropy(torch.log_softmax(logits, dim=1), target, ignore_index)
    return loss


def pad_list(xs: List[torch.Tensor], pad_value: int):

    batch_size = len(xs)
    max_len = max([x.size(0) for x in xs])
    pad = torch.zeros(batch_size, max_len, dtype=xs[0].dtype, device=xs[0].device)
    pad = pad.fill_(pad_value)
    for i in range(batch_size):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)

    ys = [y[y != ignore_id] for y in ys_pad]
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)





