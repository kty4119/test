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
  ngpus_per_node = torch.cuda.device_count()
  writer = SummaryWriter(args.log_dir)
  actual_step = (epoch + 1) * args.steps_per_epoch

  def run_validate(loader, base_progress=0):
    with torch.no_grad():
      all_image_features = []
      all_pos_text_features = []
      all_neg_text_features = []
      for i, (_, images, _, pos_tokens, pos_caption_len, _, neg_tokens, neg_caption_len) in tqdm.tqdm(enumerate(loader), position=0, total=len(loader)):
        i = base_progress + i

        if torch.cuda.is_available():
          pos_tokens = pos_tokens.cuda(args.gpu, non_blocking=True)
          pos_caption_len = pos_caption_len.cuda(args.gpu, non_blocking=True)
          neg_tokens = neg_tokens.cuda(args.gpu, non_blocking=True)
          neg_caption_len = neg_caption_len.cuda(args.gpu, non_blocking=True)
          images = images.cuda()

        if args.precision == 'fp16':
          images = images.half()
        elif args.precision == 'bf16':
          images = images.bfloat16()

        (cap_output, visual_output, pos_cap_embs, neg_cap_embs, visual_embs) = model(images, pos_tokens, pos_caption_len, neg_tokens, neg_caption_len)  # (N, T, C)
        # print(cap_output.loss)
        loss = cap_output.loss
        if args.distributed:
            original_pos_cap_embs = torch.clone(pos_cap_embs)
            original_neg_cap_embs = torch.clone(neg_cap_embs)
            all_visual_embs = [torch.zeros_like(visual_embs) for _ in range(dist.get_world_size())]
            all_pos_cap_embs = [torch.zeros_like(pos_cap_embs) for _ in range(dist.get_world_size())]
            all_neg_cap_embs = [torch.zeros_like(pos_cap_embs) for _ in range(dist.get_world_size())]
            
            dist.all_gather(all_visual_embs, visual_embs)
            dist.all_gather(all_pos_cap_embs, pos_cap_embs)
            dist.all_gather(all_neg_cap_embs, neg_cap_embs)

            # Overwrite with embeddings produced on this replica, which track the gradients.
            all_visual_embs[dist.get_rank()] = visual_embs
            all_pos_cap_embs[dist.get_rank()] = pos_cap_embs
            all_neg_cap_embs[dist.get_rank()] = neg_cap_embs
            visual_embs = torch.cat(all_visual_embs)
            pos_cap_embs = torch.cat(all_pos_cap_embs)
            neg_cap_embs = torch.cat(all_neg_cap_embs)
            start_idx = args.rank * images.shape[0]
            end_idx = start_idx + images.shape[0]
            assert torch.all(pos_cap_embs[start_idx:end_idx] == original_pos_cap_embs), args.rank

        all_pos_text_features.append(pos_cap_embs.cpu())
        all_neg_text_features.append(neg_cap_embs.cpu())
        all_image_features.append(visual_embs.cpu())

        if i % args.print_freq == 0:
          progress.display(i + 1)

        if i == args.val_steps_per_epoch - 1:
          break

      # Measure retrieval metrics over the entire validation set.
      all_image_features = torch.cat(all_image_features, axis=0)
      all_pos_text_features = torch.cat(all_pos_text_features, axis=0)
      all_neg_text_features = torch.cat(all_neg_text_features, axis=0)

      print(f"Computing similarity between {all_pos_text_features.shape} and {all_neg_text_features.shape}.")
      logits_per_pos_text = all_pos_text_features @ all_image_features.t()
      logits_per_neg_text = all_neg_text_features @ all_image_features.t()
      metrics = losses_utils.sugar_crepe_acc(logits_per_pos_text, logits_per_neg_text)
      print("metrics: ", metrics)
      
      all_image_acc1, all_image_acc5 = losses_utils.contrastive_acc(logits_per_pos_text, topk=(1, 5))
      all_caption_acc1, all_caption_acc5 = losses_utils.contrastive_acc(logits_per_neg_text, topk=(1, 5))
      image_loss = losses_utils.contrastive_loss(logits_per_pos_text)
      caption_loss = losses_utils.contrastive_loss(logits_per_neg_text)
      print("all_image_acc1: ", all_image_acc1)
      loss = args.loss_scale * (image_loss + caption_loss) / 2.0
      cont_losses.update(loss.item(), logits_per_pos_text.size(0))
      top1_caption.update(all_caption_acc1.item(), logits_per_pos_text.size(0))
      top5_caption.update(all_caption_acc5.item(), logits_per_pos_text.size(0))
      top1_image.update(all_image_acc1.item(), logits_per_pos_text.size(0))
      top5_image.update(all_image_acc5.item(), logits_per_pos_text.size(0))
      # metrics.update(metrics.item(), logits_per_pos_text.size(0))

  cont_losses = utils.AverageMeter('ContLoss', ':.4e', utils.Summary.AVERAGE)
  top1_caption = utils.AverageMeter('CaptionAcc@1', ':6.2f', utils.Summary.AVERAGE)
  top5_caption = utils.AverageMeter('CaptionAcc@5', ':6.2f', utils.Summary.AVERAGE)
  top1_image = utils.AverageMeter('ImageAcc@1', ':6.2f', utils.Summary.AVERAGE)
  top5_image = utils.AverageMeter('ImageAcc@5', ':6.2f', utils.Summary.AVERAGE)
  metrics = utils.AverageMeter('Metrics', ':6.2f', utils.Summary.AVERAGE)


  progress = utils.ProgressMeter(
    len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
    [cont_losses,  top1_caption, top5_caption, top1_image, top5_image, metrics],
    prefix='Test: ')

  # switch to evaluate mode
  model.eval()

  run_validate(val_loader)
  if args.distributed:
    cont_losses.all_reduce()
    top1_caption.all_reduce()
    top5_caption.all_reduce()
    top1_image.all_reduce()
    top5_image.all_reduce()
    metrics.all_reduce()

  if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
    aux_val_dataset = Subset(val_loader.dataset,
                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
    aux_val_loader = torch.utils.data.DataLoader(
      aux_val_dataset, batch_size=(args.val_batch_size or args.batch_size), shuffle=False,
      num_workers=args.workers, pin_memory=True, collate_fn=data.collate_fn)
    run_validate(aux_val_loader, len(val_loader))

  progress.display_summary()
  writer.add_scalar('val/contrastive_loss', cont_losses.avg, actual_step)
  writer.add_scalar('val/t2i_top1_acc', top1_caption.avg, actual_step)
  writer.add_scalar('val/t2i_top5_acc', top5_caption.avg, actual_step)
  writer.add_scalar('val/i2t_top1_acc', top1_image.avg, actual_step)
  writer.add_scalar('val/i2t_top5_acc', top5_image.avg, actual_step)
  writer.add_scalar('val/top1_acc', (top1_caption.avg + top1_image.avg) / 2.0, actual_step)
  writer.add_scalar('val/top5_acc', (top5_caption.avg + top5_image.avg) / 2.0, actual_step)
  writer.add_scalar('val/metrics', metrics, actual_step)

  writer.close()

  # Use top1 accuracy as the metric for keeping the best checkpoint.
  return metrics
  # return top1_caption.avg