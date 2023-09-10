# Copyright (c) 2023 Chuanze Lu
import argparse
import copy
import os
import yaml
import logging
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from dataset.dataset import TextDataset
from dataset.dataset import read_symbol_table
from model import create_model, criterion, add_sos_eos
from checkpoint import save_checkpoint



def get_args():
    parser = argparse.ArgumentParser(description='Do Training...')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--dev_data', required=True, help= 'cross validate data file')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pinned memory buffers used for reading')
    parser.add_argument('--num_workers', type=int, default=0, help='num of subprocess workers for processing data')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table', required=True, help='model unit symbol')
    parser.add_argument('--checkpoint', default=None, help='checkpoint model')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')

    args = parser.parse_args()
    return args


class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, data_loader, device, epoch, configs):
        ''' Train one epoch
        '''
        model.train()

        sos = configs['dataset_conf']['sos_id']
        eos = configs['dataset_conf']['eos_id']
        clip = configs.get('grad_clip', 50.0)
        log_interval = configs.get('log_interval', 10)
        epoch = epoch


        num_total_batch = 0
        total_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            key, target, target_lengths = batch
            target = target.to(device)
            target_lengths = target_lengths.to(device)
            num_utts = target_lengths.size(0)
            if num_utts == 0:
                continue

            src, dest = add_sos_eos(target, sos, eos, -1)
            target_lengths = target_lengths + 1
            logits = model(src, target_lengths)
            loss = criterion(logits, dest)

            optimizer.zero_grad()
            loss.backward()
            # grad_norm = clip_grad_norm_(model.parameters(), clip)
            # if torch.isfinite(grad_norm):
            #     optimizer.step()
            optimizer.step()
            if batch_idx % log_interval == 0:
                logging.debug(
                    'TRAIN Batch {}/{} loss {:.8f} '.format(
                        epoch, batch_idx, loss.item()))

    def cv(self, model, data_loader, device,  epoch, configs):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = configs.get('log_interval', 10)
        epoch = epoch
        # in order to avoid division by 0
        num_seen_utts = 0
        total_loss = 0.0

        sos = configs['dataset_conf']['sos_id']
        eos = configs['dataset_conf']['eos_id']

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, target, target_lengths = batch

                target = target.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                src, dest = add_sos_eos(target, sos, eos, -1)
                target_lengths = target_lengths + 1
                logits = model(src, target_lengths)
                loss = criterion(logits, dest)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    logging.debug(
                        'CV Batch {}/{} loss {:.8f} history loss {:.8f}'
                        .format(epoch, batch_idx, loss.item(),
                                total_loss / num_seen_utts))
        print(num_seen_utts)
        return total_loss / num_seen_utts

    def test(self, model, data_loader, device, args):
        return self.cv(model, data_loader, device, args)


def main():
    args = get_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    distributed = args.world_size > 1
    torch.manual_seed(7777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    symbol_table = read_symbol_table(args.symbol_table)

    train_conf = configs.get('dataset_conf', {})

    cv_conf = copy.deepcopy(train_conf)
    cv_conf['shuffle'] = False

    train_dataset = TextDataset(distributed, symbol_table, args.train_data,  **train_conf)

    cv_dataset = TextDataset(distributed, symbol_table, args.dev_data, **cv_conf)

    train_data_loader = train_dataset.get_loader(args.pin_memory, args.num_workers)

    cv_data_loader = cv_dataset.get_loader(args.pin_memory, args.num_workers)

    vocab_size = len(symbol_table)
    model_conf = configs.get('model_conf', {})
    model_conf['vocab_size'] = vocab_size
    configs['dataset_conf']['sos_id'] = vocab_size - 2
    configs['dataset_conf']['eos_id'] = vocab_size - 1

    model = create_model(model_conf)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {}'.format(num_params))

    if args.rank == 0:
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(args.model_dir, 'init.zip'))

    start_epoch = 0
    cv_loss = 0.0
    step = -1

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir

    if distributed:
        assert (torch.cuda.is_available())
        device = torch.device('cuda' if use_gpu else 'cpu')
        pass
    else:
        use_gpu = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_gpu else 'cpu')
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        threshold=0.01
    )

    executor = Executor()

    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.96, last_epoch=-1)
        logging.info('Epoch {} train info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, train_data_loader, device, epoch, configs)
        cv_loss = executor.cv(model, cv_data_loader, device, epoch, configs)
        scheduler.step()
        print(cv_loss)
        if args.rank == 0:
            save_model_path = os.path.join(args.model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model,
                save_model_path,
                {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss
                }
            )


if __name__ == '__main__':
    main()

