from typing import Optional, Tuple, List

import collections
import logging
import os
import numpy as np
import pandas as pd
import json
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset

from ic import utils

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args, split: str, tokenizer, precision: str = 'fp32') -> Dataset:
  assert split in ['train', 'val'
    ], 'Expected split to be one of "train" or "val", got {split} instead.'

  dataset_paths = []
  image_data_dirs = []
  train = split == 'train'

  # Default configs for datasets.
  # Folder structure should look like:
  if split == 'train':
    if 'coco' in args.dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'captions_train2014.json'))
      image_data_dirs.append(os.path.join(args.image_dir, 'train2014'))
    else:
      raise NotImplementedError

  elif split == 'val':
    if 'coco' in args.val_dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'captions_val2014.json'))
      image_data_dirs.append(os.path.join(args.image_dir, 'val2014'))
    else:
      raise NotImplementedError

    assert len(dataset_paths) == len(image_data_dirs) == 1, (dataset_paths, image_data_dirs)
  else:
    raise NotImplementedError

  if len(dataset_paths) > 1:
    print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
    dataset = torch.utils.data.ConcatDataset([
      JsonDataset(path, image_dir, tokenizer, 'annotations', 'image_id',
        'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1:
    dataset = JsonDataset(dataset_paths[0], image_data_dirs[0], tokenizer, 'annotations', 'image_id',
      'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
      image_size=args.image_size)
  else:
    raise ValueError(f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
  return dataset


class JsonDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, tokenizer, annotation ,img_key,
               caption_key, feature_extractor_model: str,
               train: bool = True, max_len: int = 16, sep="\t", precision: str = 'fp32',
               image_size: int = 224, num_clip_tokens: int = 1):
    logging.debug(f'Loading json data from {input_filename}.')
    with open(input_filename, 'r') as file:
      df = json.load(file)

    self.base_image_dir = base_image_dir
    self.images = utils.format_to_12_digits([item[img_key] for item in df[annotation]])
    self.captions = [item[caption_key] for item in df[annotation]]
    assert len(self.images) == len(self.captions)

    self.feature_extractor_model = feature_extractor_model
    self.feature_extractor = utils.get_feature_extractor_for_model(
      feature_extractor_model, image_size=image_size, train=False)
    self.image_size = image_size

    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision
    self.num_clip_tokens = num_clip_tokens

    self.font = None

    logging.debug('Done loading data.')

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):
    while True:
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      caption = str(self.captions[idx])
      # print(caption)
    #   clip_l_path = os.path.join(self.base_image_dir, 'clip_embs', str(self.images[idx]) + '.npy')

      try:
        img = Image.open(image_path)
        images = utils.get_pixel_values_for_model(self.feature_extractor, img)

        # Generation mode.
        caption = caption
        tokenized_data = self.tokenizer(
          caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        tokens = tokenized_data.input_ids[0]
        caption_len = tokenized_data.attention_mask[0].sum()

        decode_caption = self.tokenizer.decode(tokens, skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        cap_img = utils.create_image_of_text(decode_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)

        return image_path, images, cap_img, tokens, caption_len
      except Exception as e:
        print(f'Error reading for {image_path} with caption {caption}: {e}')
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)