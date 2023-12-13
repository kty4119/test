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
  token_idx: List[int] = [0]
  num_tokens: int = 8
  
class ICModel(nn.Module):
  def __init__(self, tokenizer, args: ICArgs = ICArgs()):
    super().__init__()
    self.tokenizer = tokenizer
    self.feature_extractor = utils.get_feature_extractor_for_model(args.visual_encoder, train=False)
    self.image_token = self.tokenizer.cls_token_id
    self.args = args
    self.num_tokens = args.num_tokens
    self.emb_dim = 256
    # self.c_h_in_dim = 17 * 4096
    # self.v_h_in_dim = 7 * 4096
    # self.v_emb_dim = 7 * 4096
    # self.v_in_dim = 196 * 768

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
      
    self.token_idx = args.token_idx
    self.lm.resize_token_embeddings(len(tokenizer))

    self.input_embeddings = self.lm.get_input_embeddings()

    print("Restoring pretrained weights for the visual model.")
    if 'vit' in visual_encoder:
      self.visual_model = ViTModel.from_pretrained(visual_encoder)
      self.visual_input_embeddings = self.visual_model.get_input_embeddings()
    else:
      self.visual_model = AutoModel.from_pretrained(visual_encoder)
      self.visual_input_embeddings = self.visual_model.get_input_embeddings()

    hidden_size = self.visual_model.config.hidden_size
    
    if self.args.freeze_vm:
      print("Freezing the VM.")
      self.visual_model.eval()
      for param in self.visual_model.parameters():
        param.requires_grad = False
    else:
      self.visual_model.train()

    self.visual_model_name = visual_encoder
    
    embedding_dim = self.input_embeddings.embedding_dim * 32 # 4096 * 17
    self.cap_hidden_fcs = nn.ModuleList([])
    self.visual_hidden_fcs = nn.ModuleList([])
    
    for layer_idx in self.args.text_emb_layers:
      if (layer_idx == -1 or layer_idx == self.lm.config.num_hidden_layers) and ('bert' not in opt_version):
        in_dim = 4096
        
        self.cap_hidden_fcs.append(
            layer.Adapter(in_dim, self.emb_dim, num_input_tokens=self.args.num_tokens,
                              num_output_tokens=1))
        self.visual_hidden_fcs.append(
            layer.Adapter(in_dim, self.emb_dim, num_input_tokens=self.args.num_tokens,
                              num_output_tokens=1))
      elif layer_idx < self.lm.config.num_hidden_layers:
        self.cap_hidden_fcs.append(
            layer.Adapter(self.lm.config.hidden_size, self.emb_dim, 
                              num_input_tokens=self.args.num_tokens, num_output_tokens=1))
        self.visual_hidden_fcs.append(
            layer.Adapter(self.lm.config.hidden_size, self.emb_dim, 
                              num_input_tokens=self.args.num_tokens, num_output_tokens=1))
      else:
        raise ValueError(f'Embedding of layer {layer_idx} was requested but model only has {self.lm.config.num_hidden_layers} layers.')
    self.visual_embeddings = nn.Linear(hidden_size, embedding_dim) # (768, 17 * 4096)
    # self.freeze_layer(self.visual_embeddings)
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
  def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

  def get_visual_embs(self, pixel_values: torch.FloatTensor):
    # Extract visual embeddings from the vision encoder.
    if 'vit' in self.visual_model_name:
      outputs = self.visual_model(pixel_values)
      outputs = outputs.pooler_output
      visual_embs = self.visual_embeddings(outputs)
      # visual_embs = visual_embs.view(visual_embs.shape[0], 17, 4096)
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 32, 4096))
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
    labels: Optional[torch.LongTensor] = None,
    caption_len: Optional[torch.LongTensor] = None
  ):
    visual_embs = self.get_visual_embs(pixel_values)

    batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens
    if labels is not None:
      assert labels.shape[0] == batch_size, (visual_embs.shape, labels.shape)

    input_embs = self.input_embeddings(labels)  # (N, T, D)
    
    cap_embedding_idx = caption_len - 1
    ### LLM에 넣기
    cap_output = self.lm(inputs_embeds=input_embs,
                       output_hidden_states=True)

    visual_output = self.lm(inputs_embeds=visual_embs,
                       output_hidden_states=True)
    ### Adapter 생성
    cap_hidden_fcs = self.cap_hidden_fcs
    visual_hidden_fcs = self.visual_hidden_fcs
    
    ### output 생성
    cap_llm_hidden_states = []
    cap_hidden_states = []
    num_tokens = self.num_tokens
    for idx, fc_layer in zip(self.args.text_emb_layers, cap_hidden_fcs):
      input_hidden_state = torch.stack([cap_output.hidden_states[idx][i, cap_embedding_idx[i]-num_tokens+1:cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
      cap_llm_hidden_states.append(input_hidden_state)
      cap_hidden_states.append(fc_layer(input_hidden_state))  # (N, seq_len, 2048)
    
    visual_llm_hidden_states = []
    visual_hidden_states = []
    for idx, fc_layer in zip(self.args.text_emb_layers, visual_hidden_fcs):
      input_hidden_state = torch.stack([visual_output.hidden_states[idx][i, cap_embedding_idx[i]-num_tokens+1:cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
      visual_llm_hidden_states.append(input_hidden_state)
      visual_hidden_states.append(fc_layer(input_hidden_state))  # (N, seq_len, 2048)
    
    cap_embedding = torch.stack(cap_hidden_states, dim=-1).sum(dim=-1)
    visual_embedding = torch.stack(visual_hidden_states, dim=-1).sum(dim=-1)
    
    visual_embs = visual_embedding[:, 0, :]
    visual_embs = visual_embs / visual_embs.norm(dim=1, keepdim=True)
    last_embedding = cap_embedding[:, 0, :]
    last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale = logit_scale.exp()
    visual_embs = logit_scale * visual_embs

    return last_embedding, visual_embs

  
class IC(nn.Module):
  def __init__(self, tokenizer, model_args: Optional[ICArgs] = None):
    super().__init__()
    self.model = ICModel(tokenizer, model_args)


  def __call__(self, images: Tensor, tgt_tokens: Optional[Tensor] = None, 
               caption_len: Optional[Tensor] = None) -> Tensor:
      output = self.model(
        pixel_values = images,
        labels = tgt_tokens,
        caption_len = caption_len)
      return output