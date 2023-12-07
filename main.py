import argparse
from collections import OrderedDict
import json
import os
import random
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"]= "5,7"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torchvision
from transformers import AutoTokenizer

from instructionClip import utils
from instructionClip import data

llm_models = ['facebook/opt-6.7b']
datasets = ['coco']
best_acc1 = 0  # Variable to keep track of best model so far.


def parse_args(args):
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--opt-version', default='facebook/opt-6.7b',
                      choices=llm_models,
                      help='OPT versions: ' +
                        ' | '.join(llm_models) +
                        ' (default: "facebook/opt-6.7b")')
    parser.add_argument('--visual-model', default='google/vit-base-patch16-224', type=str,
                      help="Visual encoder to use.")
    parser.add_argument('--num-clip-tokens', default=77, type=int, metavar='N', help='Number of CLIP token to use for generation.') # 수정 필요

    parser.add_argument('-d', '--dataset', metavar='DATASET',  help='Delimited list of datasets:' +
                      ' | '.join(datasets), default='train2014',
                      type=lambda s: [x for x in s.split(',')])

    parser.add_argument('--val-dataset', metavar='DATASET', default='val2014',
                type=lambda s: [x for x in s.split(',')],
                help='Validation dataset: ' +
                ' | '.join(datasets) +
                ' (default: coco)')
    parser.add_argument('--dataset-dir', default='/home/kty4119/coco/annotations/', type=str,
                help='Dataset directory containing .json files.')
    parser.add_argument('--image-dir', default='/home/kty4119/coco/', type=str,
                help='Dataset directory containing image folders.')
    parser.add_argument('--log-base-dir', default='./runs', type=str,
                help='Base directory to write logs and ckpts to.')
    parser.add_argument('--exp-name', default='frozen', type=str,
                help='Name of experiment, used for saving checkpoints.')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                help='number of total epochs to run')
    parser.add_argument('--steps_per_epoch', default=2000, type=int, metavar='N',
                help='number of training steps per epoch')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                help='manual epoch number (useful on restarts)')
    parser.add_argument('--val_steps_per_epoch', default=-1, type=int, metavar='N',
                help='number of validation steps per epoch')
    parser.add_argument('-b', '--batch-size', default=200, type=int,
                metavar='N',
                help='mini-batch size (default: 200), this is the total '
                'batch size of all GPUs on the current node when '
                'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--val-batch-size', default=None, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-warmup-steps', default=2000, type=int,
                metavar='N', help='Number of steps to warm up lr.')
    parser.add_argument('--lr_schedule_step_size', default=5, type=int,
                metavar='N', help='Number of steps before decaying lr.')
    parser.add_argument('--lr_schedule_gamma', default=0.1, type=float,
                metavar='N', help='Decay parameter for learning rate scheduler.')
    parser.add_argument('--grad-accumulation-steps', default=1, type=int, metavar='N',
                        help='number of gradient accumulation steps')
    parser.add_argument('--grad-clip', default=1.0, type=float, help='gradient clipping amount')

    parser.add_argument('--precision', default='bf16', type=str, choices=['fp32', 'fp16', 'bf16'],
                        help="What precision to train in.")
    parser.add_argument('--ret-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")

    parser.add_argument('--image-size', default=224, type=int, metavar='N', help='Size of images.')
    parser.add_argument('--ret-emb-dim', default=256, type=int, metavar='N', help='Embedding dimension for retrieval.')
    
    text_fc_modes = ['linear', 'gill_mapper']
    parser.add_argument('--text-fc-mode', default='gill_mapper',
                choices=text_fc_modes, help='What kind of translation mapping to use.')
    parser.add_argument('--ret-text-fc-mode', default='linear',
                choices=text_fc_modes, help='What kind of translation mapping to use.')

    parser.add_argument('--max-len', default=32, type=int,
                metavar='N', help='Maximum length to truncate captions / generations to.')
    parser.add_argument('--n-visual-tokens', default=4, type=int,
                metavar='N', help='Number of visual tokens to use for the Frozen model.')


    parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.95, type=float, metavar='M',
                help='beta2 for Adam')
    parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
                metavar='W', help='weight decay (default: 0.01)',
                dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:1337', type=str,
                help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                help='Use multi-processing distributed training to launch '
                'N processes per node, which has N GPUs. This is the '
                'fastest way to use PyTorch for either single node or '
                'multi node data parallel training')
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    i = 1
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    while os.path.exists(args.log_dir):
        args.log_dir = os.path.join(args.log_base_dir, f'{args.exp_name}_{i}')
        i += 1
    os.makedirs(args.log_dir)

    with open(os.path.join(args.log_dir, f'args.json'), 'w') as wf:
        json.dump(vars(args), wf, indent=4)

    with open(os.path.join(args.log_dir, f'git_info.txt'), 'w') as wf:
        utils.dump_git_status(out_file=wf)

    print(f'Logging to {args.log_dir}.')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                    world_size=args.world_size, rank=args.rank)
        
    # Create model
    
    
    
    
    ###
    tokenizer = AutoTokenizer.from_pretrained(args.opt_version, use_fast=False)
    if tokenizer.pad_token is None:
        if args.opt_version in ['EleutherAI/gpt-j-6B']:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("tokenizer.pad_token, tokenizer.eos_token:", tokenizer.pad_token, tokenizer.eos_token)
    # Add an image token for loss masking (and visualization) purposes.
    tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer

    
    # Data loading code
    train_dataset = data.get_dataset(args, 'train', tokenizer)
    val_dataset = data.get_dataset(args, 'val', tokenizer)
    print(f'Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=(args.val_batch_size or args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # if args.evaluate:
    #     validate.validate(val_loader, model, tokenizer, criterion, epoch, args)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        # if epoch == 0:
        #     validate.validate(val_loader, model, tokenizer, criterion, epoch-1, args)
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, tokenizer, epoch, args)
        # train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args)

    #     # evaluate on validation set
    #     acc1 = validate.validate(val_loader, model, tokenizer, criterion, epoch, args)

    #     # remember best acc@1 and save checkpoint
    #     is_best = acc1 > best_acc1
    #     best_acc1 = max(acc1, best_acc1)

    #     if not args.multiprocessing_distributed or (args.multiprocessing_distributed
    #         and args.rank % ngpus_per_node == 0):

    #         # Only save non-frozen parameters.
    #         stripped_state_dict = {
    #         k: v for k, v in model.state_dict().items() if 
    #         ('.lm' not in k and '.visual_model' not in k)
    #         }
    #     stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))
    #     utils.save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': stripped_state_dict,
    #         'best_acc1': best_acc1,
    #         'optimizer' : optimizer.state_dict(),
    #         'scheduler' : scheduler.state_dict()
    #     }, is_best, os.path.join(args.log_dir, 'ckpt'))
    
# def train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args):
def train(train_loader, tokenizer, epoch, args):
    ngpus_per_node = torch.cuda.device_count()
    
    # switch to train mode
    # model.train()
    end = time.time()
    
    for i, (_, images, cap_img, tokens, caption_len, tokens, caption_len) in enumerate(train_loader):
        actual_step = epoch * args.steps_per_epoch + i + 1
        # measure data loading time
        # data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            ret_tokens = ret_tokens.cuda(args.gpu, non_blocking=True)
            ret_caption_len = ret_caption_len.cuda(args.gpu, non_blocking=True)
            gen_tokens = gen_tokens.cuda(args.gpu, non_blocking=True)
            gen_caption_len = gen_caption_len.cuda(args.gpu, non_blocking=True)
            clip_emb = clip_emb.cuda(args.gpu, non_blocking=True)

        if args.precision == 'fp16':
            images = images.half()
        elif args.precision == 'bf16':
            images = images.bfloat16()
    
    
# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
  main(sys.argv[1:])