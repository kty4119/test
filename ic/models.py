from typing import List, Optional
from collections import namedtuple
import json
import numpy as np
import os
import glob
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from PIL import Image, UnidentifiedImageError
from requests.exceptions import ConnectionError

from transformers import AutoTokenizer, AutoModel, ViTModel, OPTForCausalLM, AutoModelForCausalLM
from ic import utils
from ic import layer

class ICArgs:
  freeze_lm: bool = True
  freeze_vm: bool = True
  opt_version: str = 'facebook/opt-6.7b'
  visual_encoder: str = 'google/vit-base-patch16-224'
  text_emb_layers: List[int] = [-1]
  
class ICModel(nn.Module):
  def __init__(self, tokenizer, args: ICArgs = ICArgs()):
    super().__init__()
    self.tokenizer = tokenizer
    self.feature_extractor = utils.get_feature_extractor_for_model(args.visual_encoder, train=False)
    self.image_token = self.tokenizer.cls_token_id
    self.args = args
    self.emb_dim = 256
    self.in_dim = 12 * 50266
    # self.in_dim = 17 * 32001
    self.v_emb_dim = 12 * 4096
    self.v_in_dim = 196 * 768

    opt_version = args.opt_version
    visual_encoder = args.visual_encoder
    print(f"Using {opt_version} for the language model.")
    print(f"Using {visual_encoder} for the visual model.")

    # if 'facebook/opt' in opt_version:
    #   self.lm = OPTForCausalLM.from_pretrained(opt_version)
    # else:
    #   raise NotImplementedError
    
    self.lm = AutoModelForCausalLM.from_pretrained(opt_version)
    self.opt_version = opt_version

    if self.args.freeze_lm:
      self.lm.eval()
      print("Freezing the LM.")
      for param in self.lm.parameters():
        param.requires_grad = False
    else:
      self.lm.train()

    self.lm.resize_token_embeddings(len(tokenizer))

    self.input_embeddings = self.lm.get_input_embeddings()

    print("Restoring pretrained weights for the visual model.")
    if 'vit' in visual_encoder:
      self.visual_model = ViTModel.from_pretrained(visual_encoder)
      self.visual_input_embeddings = self.visual_model.get_input_embeddings()
    else:
      self.visual_model = AutoModel.from_pretrained(visual_encoder)
      self.visual_input_embeddings = self.visual_model.get_input_embeddings()

    if self.args.freeze_vm:
      print("Freezing the VM.")
      self.visual_model.eval()
      for param in self.visual_model.parameters():
        param.requires_grad = False
    else:
      self.visual_model.train()

    self.visual_model_name = visual_encoder

    self.cap_hidden_fcs = nn.Linear(self.in_dim, self.emb_dim)
    self.visual_hidden_fcs = nn.Linear(self.in_dim, self.emb_dim)
    self.visual_embeddings = nn.Linear(self.v_in_dim, self.v_emb_dim)
    self.freeze_layer(self.visual_embeddings)
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
  def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

  def get_visual_embs(self, pixel_values: torch.FloatTensor):
    # Extract visual embeddings from the vision encoder.
    if 'vit' in self.visual_model_name:
      outputs = self.visual_input_embeddings(pixel_values)
      visual_embs = self.visual_embeddings(outputs.reshape(10, -1))
      visual_embs = visual_embs.view(10, 12, 4096)
    else:
      raise NotImplementedError
    return visual_embs


  def train(self, mode=True):
    super(ICModel, self).train(mode=mode)
    # Overwrite train() to ensure frozen models remain frozen.
    if self.args.freeze_lm:
      self.lm.eval()
    if self.args.freeze_vm:
      self.visual_model.eval()


  def forward(
    self,
    pixel_values: torch.FloatTensor,
    labels: Optional[torch.LongTensor] = None
  ):
    visual_embs = self.get_visual_embs(pixel_values)

    batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens
    if labels is not None:
      assert labels.shape[0] == batch_size, (visual_embs.shape, labels.shape)

    input_embs = self.input_embeddings(labels)  # (N, T, D)
    
    cap_output = self.lm(inputs_embeds=input_embs,
                       output_hidden_states=True)

    visual_output = self.lm(inputs_embeds=visual_embs,
                       output_hidden_states=True)
    
    cap_hidden_fcs = self.cap_hidden_fcs
    cap_output = cap_hidden_fcs(cap_output.logits.reshape(10, -1))
    
    visual_hidden_fcs = self.visual_hidden_fcs
    visual_output = visual_hidden_fcs(visual_output.logits.reshape(10, -1))

    return cap_output, visual_output

  
class IC(nn.Module):
  def __init__(self, tokenizer, model_args: Optional[ICArgs] = None):
    super().__init__()
    self.model = ICModel(tokenizer, model_args)


  def __call__(self, images: Tensor, tgt_tokens: Optional[Tensor] = None) -> Tensor:
      output = self.model(
        pixel_values = images,
        labels = tgt_tokens)
      return output