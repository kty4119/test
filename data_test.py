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
# dataset_paths = {
# 'add_obj'    : '/home/kty4119/coco/annotations/add_obj.json',
# 'add_att'    : '/home/kty4119/coco/annotations/add_att.json',
# 'replace_obj': '/home/kty4119/coco/annotations/replace_obj.json',
# 'replace_att': '/home/kty4119/coco/annotations/replace_att.json',
# 'replace_rel': '/home/kty4119/coco/annotations/replace_rel.json',
# 'swap_obj'   : '/home/kty4119/coco/annotations/swap_obj.json',
# 'swap_att'   : '/home/kty4119/coco/annotations/swap_att.json'
# }
# dataset = {}
# image_data_dirs=[]
# for c, data_path in dataset_paths.items():
#     dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))
# print(dataset_paths)
# print(len(dataset))
# image_data_dirs.append(os.path.join(args.image_dir, 'val2017'))
# print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
# dataset = torch.utils.data.ConcatDataset([
#     JsonDataset(path, image_dir, tokenizer, 'annotations', 'image_id',
#     'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
#     image_size=args.image_size)
#     for (path, image_dir) in zip(dataset_paths, image_data_dirs)])

json_path = '/home/kty4119/coco/annotations/add_obj.json'
# json_path = '/home/kty4119/coco/annotations/captions_train2014.json'
with open(json_path, 'r') as file:
  df = json.load(file)
# images = [item['image_id'] for item in df['annotations']]
# captions = [item['caption'] for item in df['annotations']]
images = [value['filename'] for _, value in df.items()]
pos_captions = [value['caption'] for _, value in df.items()]
neg_captions = [value['negative_caption'] for _, value in df.items()]
print(images)