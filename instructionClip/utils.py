from enum import Enum
import subprocess
import sys
import shutil
import torch
import torch.distributed as dist
from torchvision.transforms import functional as F
from torchvision import transforms as T
from transformers import AutoFeatureExtractor
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
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return feature_extractor


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