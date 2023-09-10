# Copyright (c) 2023 Chuanze Lu
import os

import yaml
import numpy as np
from net.model import create_model, input_tokenizer
from net.checkpoint import load_checkpoint
from net.dataset.dataset import read_symbol_table
import torch



def main():

    test_text = "我是"
    checkpoint = "exp/transformer2/2.pt"
    config_file = "conf/transformer.yaml"
    dict_file = 'data/dict/lang_char2.txt'
    use_gpu = True
    with open(config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    symbol_table = read_symbol_table(dict_file)
    vocab_size = len(symbol_table)
    model_conf = configs.get('model_conf', {})
    model_conf['vocab_size'] = vocab_size

    model = create_model(model_conf)
    load_checkpoint(model, checkpoint)
    use_cuda = use_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    sos_id = symbol_table['<sos>']
    eos_id = symbol_table['<eos>']

    x, x_len = input_tokenizer(symbol_table, test_text, device)
    loss = 1.0
    with torch.no_grad():
        loss = model.inference_loss(x, x_len, sos_id, eos_id)

    print(loss)


if __name__ == '__main__':
    main()
