from enum import Enum
import subprocess
import sys
import shutil
import torch
import torch.distributed as dist
from torchvision.transforms import functional as F
from torchvision import transforms as T
from transformers import AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import requests
from io import BytesIO

def dump_git_status(out_file=sys.stdout, exclude_file_patterns=['*.ipynb', '*.th', '*.sh', '*.txt', '*.json']):
    """Logs git status to stdout."""
    subprocess.call('git rev-parse HEAD', shell=True, stdout=out_file)
    subprocess.call('echo', shell=True, stdout=out_file)
    exclude_string = ''
    subprocess.call('git --no-pager diff -- . {}'.format(exclude_string), shell=True, stdout=out_file)
  
def get_feature_extractor_for_model(model_name: str, image_size: int = 224, train: bool = True):
    print(f'Using HuggingFace AutoFeatureExtractor for {model_name}.')
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    return image_processor


def get_pixel_values_for_model(feature_extractor, img: Image.Image):
    pixel_values = feature_extractor(img.convert('RGB'), return_tensors="pt").pixel_values[0, ...]  # (3, H, W)
    return pixel_values

def create_image_of_text(text: str, width: int = 224, nrows: int = 2, color=(255, 255, 255), font=None) -> torch.Tensor:
  """Creates a (3, nrows * 14, width) image of text.

  Returns:
    cap_img: (3, 14 * nrows, width) image of wrapped text.
  """
  height = 12
  padding = 5
  effective_width = width - 2 * padding
  # Create a black image to draw text on.
  cap_img = Image.new('RGB', (effective_width * nrows, height), color = (0, 0, 0))
  draw = ImageDraw.Draw(cap_img)
  draw.text((0, 0), text, color, font=font or ImageFont.load_default())
  cap_img = F.convert_image_dtype(F.pil_to_tensor(cap_img), torch.float32)  # (3, height, W * nrows)
  cap_img = torch.split(cap_img, effective_width, dim=-1)  # List of nrow elements of shape (3, height, W)
  cap_img = torch.cat(cap_img, dim=1)  # (3, height * nrows, W)
  # Add zero padding.
  cap_img = torch.nn.functional.pad(cap_img, [padding, padding, 0, padding])
  return cap_img

def format_to_12_digits(image_list):
    jpg_list =[]
    for number in image_list:
        # 숫자를 문자열로 변환
        str_number = str(number)
    
        # 현재 문자열의 자릿수 확인
        num_digits = len(str_number)
    
        # 부족한 자릿수만큼 0을 앞에 추가하여 12자리 문자열 생성
        formatted_number = '0' * (12 - num_digits) + str_number +'.jpg'
        jpg_list.append(formatted_number)
    
    return jpg_list

def accuracy(output, target, padding, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    if output.shape[-1] < maxk:
      print(f"[WARNING] Less than {maxk} predictions available. Using {output.shape[-1]} for topk.")

    maxk = min(maxk, output.shape[-1])
    batch_size = target.size(0)

    # Take topk along the last dimension.
    _, pred = output.topk(maxk, -1, True, True)  # (N, T, topk)

    mask = (target != padding).type(target.dtype)
    target_expand = target[..., None].expand_as(pred)
    correct = pred.eq(target_expand)
    correct = correct * mask[..., None].expand_as(correct)

    res = []
    for k in topk:
      correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / mask.sum()))
    return res

def get_params_count(model, max_name_len: int = 60):
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  return params, total_trainable_params, total_nontrainable_params


def get_params_count_str(model, max_name_len: int = 60):
  padding = 70  # Hardcoded depending on desired amount of padding and separators.
  params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  for name, param_count, shape, trainable in params:
    param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  return param_counts_text


class Summary(Enum):
  NONE = 0
  AVERAGE = 1
  SUM = 2
  COUNT = 3


class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))
    
  def display_summary(self):
    entries = [" *"]
    entries += [meter.summary() for meter in self.meters]
    print(' '.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
    self.name = name
    self.fmt = fmt
    self.summary_type = summary_type
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def all_reduce(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    self.sum, self.count = total.tolist()
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)
  
  def summary(self):
    fmtstr = ''
    if self.summary_type is Summary.NONE:
      fmtstr = ''
    elif self.summary_type is Summary.AVERAGE:
      fmtstr = '{name} {avg:.3f}'
    elif self.summary_type is Summary.SUM:
      fmtstr = '{name} {sum:.3f}'
    elif self.summary_type is Summary.COUNT:
      fmtstr = '{name} {count:.3f}'
    else:
      raise ValueError('invalid summary type %r' % self.summary_type)
    
    return fmtstr.format(**self.__dict__)