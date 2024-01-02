import argparse
from collections import OrderedDict
import json
import os
import random
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"
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

from ic import utils
# from ic import data
from ic import data_sugar_crepe
from ic import models
from ic import loss as losses_utils
# from ic import validate
from ic import validate_sugar_crepe

llm_models = ['facebook/opt-6.7b', '/home/shared/hub/models--ty--alpaca-7b-wdiff']
datasets = ['coco']
best_acc1 = 0  # Variable to keep track of best model so far.


def parse_args(args):
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--opt-version', default='/home/shared/hub/models--ty--alpaca-7b-wdiff',
                      choices=llm_models,
                      help='OPT versions: ' +
                        ' | '.join(llm_models) +
                        ' (default: "facebook/opt-6.7b")')
    parser.add_argument('--visual-model', default='google/vit-base-patch16-224-in21k', type=str,
                      help="Visual encoder to use.")
    parser.add_argument('--num-tokens', default=0, type=int, metavar='N', help='Number of [IMG] tokens to use.')
    parser.add_argument('-d', '--dataset', metavar='DATASET',  help='Delimited list of datasets:' +
                      ' | '.join(datasets), default='train2014',
                      type=lambda s: [x for x in s.split(',')])

    parser.add_argument('--val-dataset', metavar='DATASET', default='val2017',
                type=lambda s: [x for x in s.split(',')],
                help='Validation dataset: ' +
                ' | '.join(datasets) +
                ' (default: coco)')
    parser.add_argument('--dataset-dir', default='/home/kty4119/coco/annotations/', type=str,
                help='Dataset directory containing .json files.')
    parser.add_argument('--image-dir', default='/home/kty4119/coco/', type=str,
                help='Dataset directory containing image folders.')
    parser.add_argument('--log-base-dir', default='/home/kty4119/runs', type=str,
                help='Base directory to write logs and ckpts to.')
    parser.add_argument('--exp-name', default='frozen', type=str,
                help='Name of experiment, used for saving checkpoints.')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                help='number of total epochs to run')
    parser.add_argument('--steps_per_epoch', default=200, type=int, metavar='N',
                help='number of training steps per epoch')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                help='manual epoch number (useful on restarts)')
    parser.add_argument('--val_steps_per_epoch', default=-1, type=int, metavar='N',
                help='number of validation steps per epoch')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                metavar='N',
                help='mini-batch size (default: 100), this is the total '
                'batch size of all GPUs on the current node when '
                'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--val-batch-size', default=None, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-warmup-steps', default=200, type=int,
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
    parser.add_argument('--loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")

    parser.add_argument('--image-size', default=224, type=int, metavar='N', help='Size of images.')
    parser.add_argument('--emb-dim', default=256, type=int, metavar='N', help='Embedding dimension.')

    parser.add_argument('--max-len', default=32, type=int,
                metavar='N', help='Maximum length to truncate captions / generations to.')


    parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.95, type=float, metavar='M',
                help='beta2 for Adam')
    parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
                metavar='W', help='weight decay (default: 0.01)',
                dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
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
    model_args = models.ICArgs()
    model_args.opt_version = args.opt_version
    model_args.visual_encoder = args.visual_model
    model_args.text_emb_layers = [-1]
    model_args.freeze_lm = True
    model_args.freeze_vm = True
    model_args.emb_dim = args.emb_dim
    model_args.num_tokens = args.num_tokens
    
    # tokenizer = AutoTokenizer.from_pretrained(args.opt_version, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(args.opt_version)
    if tokenizer.pad_token is None:
        if args.opt_version in ['EleutherAI/gpt-j-6B']:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("tokenizer.pad_token, tokenizer.eos_token:", tokenizer.pad_token, tokenizer.eos_token)
    # Add an summary token for loss masking (and visualization) purposes.
    tokenizer.add_special_tokens({"cls_token": "<|summary|>"})  # add special summary token to tokenizer

    # Add [IMG] tokens to the vocabulary.
    model_args.token_idx = []
    args.token_idx = []
    for i in range(model_args.num_tokens):
        print(f'Adding [IMG{i}] token to vocabulary.')
        print(f'Before adding new token, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
        num_added_tokens = tokenizer.add_tokens(f'[IMG{i}]')
        print(f'After adding {num_added_tokens} new tokens, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
        token_idx = tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
        assert len(token_idx) == 1, token_idx
        model_args.token_idx.append(token_idx[0])
        args.token_idx.append(token_idx[0])

    # Save model args to disk.
    with open(os.path.join(args.log_dir, 'model_args.json'), 'w') as f:
        json.dump(vars(model_args), f, indent=4)

    model = models.IC(tokenizer, model_args)
    if args.precision == 'fp16':
        model = model.float()
    elif args.precision == 'bf16':
        model = model.bfloat16()

    # Print parameters and count of model.
    param_counts_text = utils.get_params_count_str(model)
    with open(os.path.join(args.log_dir, 'param_count.txt'), 'w') as f:
        f.write(param_counts_text)

    # Log trainable parameters to Tensorboard.
    _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)
    writer = SummaryWriter(args.log_dir)
    writer.add_scalar('params/total', total_trainable_params + total_nontrainable_params, 0)
    writer.add_scalar('params/total_trainable', total_trainable_params, 0)
    writer.add_scalar('params/total_non_trainable', total_nontrainable_params, 0)
    writer.close()

    if not torch.cuda.is_available():
        print('WARNING: using CPU, this will be slow!')
        model = torch.nn.DataParallel(model)
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.val_batch_size = int((args.val_batch_size or args.batch_size) / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer_cls = torch.optim.AdamW
    print('Using torch.optim.AdamW as the optimizer.')
    optimizer = optimizer_cls(model.parameters(), args.lr,
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay,
                    eps=1e-8)

    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    scheduler_steplr = StepLR(optimizer, step_size=args.lr_schedule_step_size * args.steps_per_epoch, gamma=args.lr_schedule_gamma)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.lr_warmup_steps, after_scheduler=scheduler_steplr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # Data loading code
    # train_dataset = data.get_dataset(args, 'train', tokenizer)
    # val_dataset = data.get_dataset(args, 'val', tokenizer)
    train_dataset = data_sugar_crepe.get_dataset(args, 'train', tokenizer)
    val_dataset = data_sugar_crepe.get_dataset(args, 'val', tokenizer)
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

    if args.evaluate:
        epoch = 0
        # validate.validate(val_loader, model, tokenizer, criterion, epoch, args)
        validate_sugar_crepe.validate(val_loader, model, tokenizer, criterion, epoch, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if epoch == 0:
            # validate.validate(val_loader, model, tokenizer, criterion, epoch-1, args)
            validate_sugar_crepe.validate(val_loader, model, tokenizer, criterion, epoch-1, args)
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args)

        # evaluate on validation set
        # acc1 = validate.validate(val_loader, model, tokenizer, criterion, epoch, args)
        acc1 = validate_sugar_crepe.validate(val_loader, model, tokenizer, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
            # Only save non-frozen parameters.
            stripped_state_dict = {
            k: v for k, v in model.state_dict().items() if 
            ('.lm' not in k and '.visual_model' not in k)
            }
            stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': stripped_state_dict,
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, os.path.join(args.log_dir, 'ckpt'))
    
def train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args):
    # ngpus_per_node = torch.cuda.device_count()
    cont_losses = utils.AverageMeter('ContLoss', ':.4e')
    losses = utils.AverageMeter('Loss', ':.4e')
    cap_ce_losses = utils.AverageMeter('CapCeLoss', ':.4e')
    # vis_ce_losses = utils.AverageMeter('VisCeLoss', ':.4e')
    top1_caption = utils.AverageMeter('AccCaption@1', ':6.2f')
    top5_caption = utils.AverageMeter('AccCaption@5', ':6.2f')
    top1_image = utils.AverageMeter('AccImage@1', ':6.2f')
    top5_image = utils.AverageMeter('AccImage@5', ':6.2f')
    
    writer = SummaryWriter(args.log_dir)
    
    progress = utils.ProgressMeter(
    args.steps_per_epoch,
    [cont_losses, top1_caption, top1_image],
    prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()
    
    for i, (_, images, cap_img, tokens, caption_len) in enumerate(train_loader):
        actual_step = epoch * args.steps_per_epoch + i + 1

        if torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            tokens = tokens.cuda(args.gpu, non_blocking=True)
            caption_len = caption_len.cuda(args.gpu, non_blocking=True)

        if args.precision == 'fp16':
            images = images.half()
        elif args.precision == 'bf16':
            images = images.bfloat16()
            
        loss = 0
            
        (cap_output, visual_output, cap_embs, visual_embs) = model(images, tokens, caption_len)
        print(cap_output.loss)
        # print(visual_output.loss)
        cap_ce_loss = cap_output.loss * 0.5
        # vis_ce_loss = visual_output.loss * 0.5
        loss += cap_ce_loss
        # loss += vis_ce_loss
        
        cap_ce_losses.update(cap_ce_loss.item(), images.size(0))
        # vis_ce_losses.update(vis_ce_loss.item(), images.size(0))
        
        if args.distributed:
          all_visual_embs = [torch.zeros_like(visual_embs) for _ in range(dist.get_world_size())]
          all_cap_embs = [torch.zeros_like(cap_embs) for _ in range(dist.get_world_size())]
          dist.all_gather(all_visual_embs, visual_embs)
          dist.all_gather(all_cap_embs, cap_embs)
          # Overwrite with embeddings produced on this replace, which have the gradient.
          all_visual_embs[dist.get_rank()] = visual_embs
          all_cap_embs[dist.get_rank()] = cap_embs
          visual_embs = torch.cat(all_visual_embs)
          cap_embs = torch.cat(all_cap_embs)

        print(visual_embs.shape, cap_embs.shape)
        logits_per_image = visual_embs @ cap_embs.t()
        logits_per_text = logits_per_image.t()
        if i == 0:
          print(f'Running contrastive loss over logits_per_text.shape = {logits_per_text.shape} and logits_per_image.shape = {logits_per_image.shape}')

        caption_loss = losses_utils.contrastive_loss(logits_per_text)
        image_loss = losses_utils.contrastive_loss(logits_per_image)
        caption_acc1, caption_acc5 = losses_utils.contrastive_acc(logits_per_text, topk=(1, 5))
        image_acc1, image_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))
        loss += args.loss_scale * (caption_loss + image_loss) / 2.0
        cont_losses.update(loss.item(), images.size(0))

        # measure accuracy and record loss
        top1_caption.update(caption_acc1[0], images.size(0))
        top5_caption.update(caption_acc5[0], images.size(0))
        top1_image.update(image_acc1[0], images.size(0))
        top5_image.update(image_acc5[0], images.size(0))
        
        loss = loss / args.grad_accumulation_steps
        losses.update(loss.item(), images.size(0))
        loss.backward()
        
        # 업데이트 안하는 파라미터 확인
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        
        # Update weights
        if ((i + 1) % args.grad_accumulation_steps == 0) or (i == args.steps_per_epoch - 1):
            for param in model.module.model.input_embeddings.parameters():
                assert param.grad.shape[0] == len(tokenizer)
                # Keep other embeddings frozen.
                mask = torch.zeros((param.grad.shape[0], 1)).to(param.grad)
                # for idx in args.token_idx:
                #     mask[idx] = 1
                param.grad = param.grad * mask

            # compute gradient and do SGD step
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            print('=' * 80)
        with torch.no_grad():
            # Normalize trainable embeddings.
            frozen_norm = torch.norm(model.module.model.input_embeddings.weight[:-args.num_tokens, :], dim=1).mean(0)
            for idx in args.token_idx:
                trainable_weight = model.module.model.input_embeddings.weight[idx, :]
                model.module.model.input_embeddings.weight[idx, :].div_(trainable_weight.norm(dim=-1) / frozen_norm)
        if actual_step == 1 or (i + 1) % args.print_freq == 0:
            print('First 5 values of first 3 tokens of embedding matrix:', model.module.model.input_embeddings.weight.data[:3, :5])
            # print('First 5 values of first [IMG0] token embeddings:', model.module.model.input_embeddings.weight.data[args.token_idx[0], :5])
            # print(f'First 5 values of first [IMG{args.num_tokens-1}] token embeddings:', model.module.model.input_embeddings.weight.data[args.token_idx[-1], :5])
            if args.distributed:
                losses.all_reduce()
                cont_losses.all_reduce()
                cap_ce_losses.all_reduce()
                # vis_ce_losses.all_reduce()
                top1_caption.all_reduce()
                top5_caption.all_reduce()
                top1_image.all_reduce()
                top5_image.all_reduce()

        progress.display(i + 1)

        writer.add_scalar('train/loss', losses.avg, actual_step)
        writer.add_scalar('train/contrastive_loss', cont_losses.avg, actual_step)
        writer.add_scalar('train/cap_ce_loss', cap_ce_losses.avg, actual_step)
        # writer.add_scalar('train/vis_ce_loss', vis_ce_losses.avg, actual_step)

        writer.add_scalar('train/t2i_top1_acc', top1_caption.avg, actual_step)
        writer.add_scalar('train/t2i_top5_acc', top5_caption.avg, actual_step)
        writer.add_scalar('train/i2t_top1_acc', top1_image.avg, actual_step)
        writer.add_scalar('train/i2t_top5_acc', top5_image.avg, actual_step)

        losses.reset()
        cont_losses.reset()
        cap_ce_losses.reset()
        # vis_ce_losses.reset()
        top1_caption.reset()
        top5_caption.reset()
        top1_image.reset()
        top5_image.reset()
    
        if i == args.steps_per_epoch - 1:
            break
        
        scheduler.step()
        curr_lr = scheduler.get_last_lr()
        if (actual_step == 1) or (i + 1) % args.print_freq == 0:
            # Write current learning rate to Tensorboard.
            writer = SummaryWriter(args.log_dir)
            writer.add_scalar('train/lr', curr_lr[0], actual_step)
            writer.close()

    writer.close()

    
# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
  main(sys.argv[1:])