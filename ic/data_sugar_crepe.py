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
      # dataset_paths = {
      #   'add_obj'    : f'{args.dataset_dir}/add_obj.json',
      #   'add_att'    : f'{args.dataset_dir}/add_att.json',
      #   'replace_obj': f'{args.dataset_dir}/replace_obj.json',
      #   'replace_att': f'{args.dataset_dir}/replace_att.json',
      #   'replace_rel': f'{args.dataset_dir}/replace_rel.json',
      #   'swap_obj'   : f'{args.dataset_dir}/swap_obj.json',
      #   'swap_att'   : f'{args.dataset_dir}/swap_att.json'
      #   }
      dataset_paths = [
        # f'{args.dataset_dir}add_obj.json',
        # f'{args.dataset_dir}add_att.json',
        f'{args.dataset_dir}replace_obj.json',
        # f'{args.dataset_dir}replace_att.json',
        # f'{args.dataset_dir}replace_rel.json',
        # f'{args.dataset_dir}swap_obj.json',
        # f'{args.dataset_dir}swap_att.json'
        ]
      # dataset = {}
      # for c, data_path in dataset_paths.items():
      #   dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))
      # dataset_paths.append(os.path.join(args.dataset_dir, 'captions_val2014_v2.json'))
      for i in range(len(dataset_paths)):
        image_data_dirs.append(os.path.join(args.image_dir, 'val2017'))
    else:
      raise NotImplementedError

    assert len(dataset_paths) == len(image_data_dirs), (dataset_paths, image_data_dirs)
  else:
    raise NotImplementedError

  if len(dataset_paths) > 1: # val
    print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
    dataset = torch.utils.data.ConcatDataset([
      Val_JsonDataset(path, image_dir, tokenizer, 'filename', 'caption',
        'negative_caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1: # train
    dataset = Val_JsonDataset(dataset_paths[0], image_data_dirs[0], tokenizer, 'filename', 'caption',
        'negative_caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
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
        
class Val_JsonDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, tokenizer, filename ,caption,
               negative_caption, feature_extractor_model: str,
               train: bool = True, max_len: int = 16, sep="\t", precision: str = 'fp32',
               image_size: int = 224, num_clip_tokens: int = 1):
    logging.debug(f'Loading json data from {input_filename}.')
    with open(input_filename, 'r') as file:
      df = json.load(file)
    
    self.base_image_dir = base_image_dir
    self.images = [value[filename] for _, value in df.items()]
    self.pos_captions = [value[caption] for _, value in df.items()]
    self.neg_captions = [value[negative_caption] for _, value in df.items()]
    # self.images = utils.format_to_12_digits([item[img_key] for item in df[annotation]])
    # self.captions = [item[caption_key] for item in df[annotation]]
    assert len(self.images) == len(self.pos_captions) == len(self.neg_captions)

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
    return len(self.pos_captions)

  def __getitem__(self, idx):
    while True:
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      pos_caption = str(self.pos_captions[idx])
      neg_caption = str(self.neg_captions[idx])
      # print(image_path)
      # print("pos_caption: ", pos_caption)
      # print("neg_caption: ", neg_caption)

      try:
        img = Image.open(image_path)
        images = utils.get_pixel_values_for_model(self.feature_extractor, img)

        # Generation mode.
        ### pos caption
        ### swap_att
        # instruction = "Focus on swaps for matching with images: " # 0.529
        # instruction = "Focus on match images with swapped attribute: " # 0.544
        # instruction = "Focus on swapped attribute for matching with images: " # 0.550
        # instruction = "### Instruction: Focus on the attributes of the object in the caption.\n### Caption: "
        
        ### swap_obj
        # instruction = "Focus on swaps for matching with images: " # 0.588
        # instruction = "Focus on match images with swapped object: " # 0.612
        # instruction = "Focus on swapped object for matching with images: " # 0.608
        # instruction = "### Instruction: Focus on the attributes of the object in the caption.\n### Caption: "
        
        ### replace_rel
        # instruction = "Focus on replaces for matching with images: " # 0.619
        # instruction = "Focus on match images with replaced relation: " # 0.639
        # instruction = "Focus on replaced relation for matching with images: " # 0.634
        # instruction = "### Instruction: Focus on the relationships between each entity in the caption.\n### Caption: "
        
        ### replace_att
        # instruction = "Focus on replaces for matching with images: " # 0.642
        # instruction = "Focus on match images with replaced attribute: " # 0.641
        # instruction = "Focus on replaced attribute for matching with images: " # 0.643
        # instruction = "Focus on 'Modifiers' in image: " # 0.643
        # instruction = "### Instruction: Focus on the attribute the object has in the caption.\n### Caption: "
        # instruction = "### Instruction: Focus on each object's properties in captions.\n### Caption: "
        
        ### replace_obj
        # instruction = "Focus on replaces for matching with images: " # 0.835
        # instruction = "Focus on match images with replaced object: " # 0.837
        # instruction = "Focus on replaced object for matching with images: " # 0.831
        # instruction = "Focus on the objects: "
        # instruction = "### Instruction: Focus on the attribute the object has in the caption.\n### Caption: "
        # no instruction: 0.837
        
        ### add_att
        # instruction = "Focus on attribute for matching with images: " # 0.543
        # instruction = "Focus on match images with added attribute: " # 0.565
        # instruction = "Focus on added attribute for matching with images: " # 0.553
        # instruction = "Focus on the 'Modifiers' in caption: " # 0.581
        # instruction = "### Instruction: Focus on the attribute the object has in the caption.\n### Caption: "
        # no instruction: 0.594
        
        ### add_obj
        # instruction = "Focus on object for matching with images: " # 0.662
        # instruction = "Focus on match images with added object: " # 0.699
        # instruction = "Focus on added object for matching with images: " # 0.690
        # instruction = "Focus on the object in image: " # 0.727
        # instruction = "### Instruction: Focus on the attribute the object has in the caption.\n### Caption: "
        # no instruction: 0.726
        pos_messages = [
          {
              "role": "system",
              "content": "You are a bot that answers according to the given instruction.",
          },
          {"role": "user", "content": "### Instruction: Find the object in the caption.\n### Caption: " + str(pos_caption)},
        ]
        
        pos_caption = self.tokenizer.apply_chat_template(pos_messages, tokenize=False, add_generation_prompt=True)
        print(pos_caption)
        # pos_caption = instruction + pos_caption
        tokenized_pos_data = self.tokenizer(
          pos_caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        pos_tokens = tokenized_pos_data.input_ids[0]
        pos_caption_len = tokenized_pos_data.attention_mask[0].sum()
        # print(pos_caption, pos_caption_len)
        decode_pos_caption = self.tokenizer.decode(pos_tokens, skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        pos_cap_img = utils.create_image_of_text(decode_pos_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)
        
        ### neg caption
        neg_messages = [
          {
              "role": "system",
              "content": "You are a bot that answers according to the given instruction.",
          },
          {"role": "user", "content": "### Instruction: Find the object in the caption.\n### Caption: " + str(neg_caption)},
        ]
        # Find the attributes of the object in the caption.
        
        
        neg_caption = self.tokenizer.apply_chat_template(neg_messages, tokenize=False, add_generation_prompt=True)
        print(neg_caption)
        # neg_caption = instruction + neg_caption
        tokenized_neg_data = self.tokenizer(
          neg_caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        neg_tokens = tokenized_neg_data.input_ids[0]
        neg_caption_len = tokenized_neg_data.attention_mask[0].sum()
        # print(neg_caption, neg_caption_len)
        decode_neg_caption = self.tokenizer.decode(neg_tokens, skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        neg_cap_img = utils.create_image_of_text(decode_neg_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)

        return image_path, images, pos_cap_img, pos_tokens, pos_caption_len, neg_cap_img, neg_tokens, neg_caption_len
      except Exception as e:
        print(f'Error reading for {image_path} with caption {pos_caption}: {e}')
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)