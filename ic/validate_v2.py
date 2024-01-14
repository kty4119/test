import collections
from PIL import Image
import time
import tqdm
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
import torchvision

from ic import loss as losses_utils
from ic import utils
from ic import data

def validate(val_loader, model, tokenizer, criterion, epoch, args):
  # ngpus_per_node = torch.cuda.device_count()
  base_progress=0
  
  writer = SummaryWriter(args.log_dir)
  
  actual_step = (epoch + 1) * args.steps_per_epoch
  cont_losses = utils.AverageMeter('ContLoss', ':.4e', utils.Summary.AVERAGE)
  top1_caption = utils.AverageMeter('CaptionAcc@1', ':6.2f', utils.Summary.AVERAGE)
  top5_caption = utils.AverageMeter('CaptionAcc@5', ':6.2f', utils.Summary.AVERAGE)
  top1_image = utils.AverageMeter('ImageAcc@1', ':6.2f', utils.Summary.AVERAGE)
  top5_image = utils.AverageMeter('ImageAcc@5', ':6.2f', utils.Summary.AVERAGE)
  
  progress = utils.ProgressMeter(
    len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
    [cont_losses,  top1_caption, top5_caption, top1_image, top5_image],
    prefix='Test: ')
  
  # switch to evaluate mode
  model.eval()

  # def run_validate(loader, base_progress=0):
  with torch.no_grad():
    all_image_features = []
    all_text_features = []
    for i, (_, images, _, tokens, caption_len) in tqdm.tqdm(enumerate(val_loader), position=0, total=len(val_loader)):
      i = base_progress + i

      if torch.cuda.is_available():
        tokens = tokens.cuda(args.gpu, non_blocking=True)
        caption_len = caption_len.cuda(args.gpu, non_blocking=True)
        images = images.cuda()

      if args.precision == 'fp16':
        images = images.half()
      elif args.precision == 'bf16':
        images = images.bfloat16()
        
      loss = 0

      (cap_output, visual_output, cap_embs, visual_embs) = model(images, tokens, caption_len)  # (N, T, C)
      print(cap_output.loss)
      loss = cap_output.loss * 0.5
      if args.distributed:
        original_cap_embs = torch.clone(cap_embs)
        all_visual_embs = [torch.zeros_like(visual_embs) for _ in range(dist.get_world_size())]
        all_cap_embs = [torch.zeros_like(cap_embs) for _ in range(dist.get_world_size())]

        dist.all_gather(all_visual_embs, visual_embs)
        dist.all_gather(all_cap_embs, cap_embs)

        # Overwrite with embeddings produced on this replica, which track the gradients.
        all_visual_embs[dist.get_rank()] = visual_embs
        all_cap_embs[dist.get_rank()] = cap_embs
        visual_embs = torch.cat(all_visual_embs)
        cap_embs = torch.cat(all_cap_embs)
        start_idx = args.rank * images.shape[0]
        end_idx = start_idx + images.shape[0]
        assert torch.all(cap_embs[start_idx:end_idx] == original_cap_embs), args.rank

      all_text_features.append(cap_embs.cpu())
      all_image_features.append(visual_embs.cpu())
      text_features = cap_embs.cpu()
      image_features = visual_embs.cpu()
      print(f"Computing similarity between {text_features.shape} and {image_features.shape}.")
      logits_per_image = image_features @ text_features.t()
      logits_per_text = logits_per_image.t()
      all_image_acc1, all_image_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))
      all_caption_acc1, all_caption_acc5 = losses_utils.contrastive_acc(logits_per_text, topk=(1, 5))
      image_loss = losses_utils.contrastive_loss(logits_per_image)
      caption_loss = losses_utils.contrastive_loss(logits_per_text)
      
      loss = args.loss_scale * (image_loss + caption_loss) / 2.0
      cont_losses.update(loss.item(), logits_per_image.size(0))
      top1_caption.update(all_caption_acc1.item(), logits_per_image.size(0))
      top5_caption.update(all_caption_acc5.item(), logits_per_image.size(0))
      top1_image.update(all_image_acc1.item(), logits_per_image.size(0))
      top5_image.update(all_image_acc5.item(), logits_per_image.size(0))

      if i % args.print_freq == 0:
        progress.display(i + 1)

      if i == args.val_steps_per_epoch - 1:
        break

    # Measure retrieval metrics over the entire validation set.
    all_image_features = torch.cat(all_image_features, axis=0)
    all_text_features = torch.cat(all_text_features, axis=0)

    print(f"Computing similarity between {all_image_features.shape} and {all_text_features.shape}.")
    logits_per_image = all_image_features @ all_text_features.t()
    logits_per_text = logits_per_image.t()
    all_image_acc1, all_image_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))
    all_caption_acc1, all_caption_acc5 = losses_utils.contrastive_acc(logits_per_text, topk=(1, 5))
    image_loss = losses_utils.contrastive_loss(logits_per_image)
    caption_loss = losses_utils.contrastive_loss(logits_per_text)
    
    loss = args.loss_scale * (image_loss + caption_loss) / 2.0
    cont_losses.update(loss.item(), logits_per_image.size(0))
    top1_caption.update(all_caption_acc1.item(), logits_per_image.size(0))
    top5_caption.update(all_caption_acc5.item(), logits_per_image.size(0))
    top1_image.update(all_image_acc1.item(), logits_per_image.size(0))
    top5_image.update(all_image_acc5.item(), logits_per_image.size(0))

  if args.distributed:
    cont_losses.all_reduce()
    top1_caption.all_reduce()
    top5_caption.all_reduce()
    top1_image.all_reduce()
    top5_image.all_reduce()

  # if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
  #   aux_val_dataset = Subset(val_loader.dataset,
  #                range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
  #   aux_val_loader = torch.utils.data.DataLoader(
  #     aux_val_dataset, batch_size=(args.val_batch_size or args.batch_size), shuffle=False,
  #     num_workers=args.workers, pin_memory=True, collate_fn=data.collate_fn)
  #   run_validate(aux_val_loader, len(val_loader))

  progress.display_summary()

  writer.add_scalar('val/contrastive_loss', cont_losses.avg, actual_step)
  writer.add_scalar('val/t2i_top1_acc', top1_caption.avg, actual_step)
  writer.add_scalar('val/t2i_top5_acc', top5_caption.avg, actual_step)
  writer.add_scalar('val/i2t_top1_acc', top1_image.avg, actual_step)
  writer.add_scalar('val/i2t_top5_acc', top5_image.avg, actual_step)
  writer.add_scalar('val/top1_acc', (top1_caption.avg + top1_image.avg) / 2.0, actual_step)
  writer.add_scalar('val/top5_acc', (top5_caption.avg + top5_image.avg) / 2.0, actual_step)

  writer.close()

  # Use top1 accuracy as the metric for keeping the best checkpoint.
  return top1_caption.avg