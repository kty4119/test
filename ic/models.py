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
# from transformers import AutoTokenizer, AutoModel, OPTForCausalLM, AutoModelForCausalLM
# from ic.custom_vit import ViTModel
from ic import utils
from ic import layer

class ICArgs:
  freeze_lm: bool = True
  freeze_vm: bool = True
  opt_version: str = '/home/shared/hub/ty_mistral_instruct'
  visual_encoder: str = 'google/vit-base-patch16-224-in21k'
  text_emb_layers: List[int] = [-1]
  token_idx: List[int] = [0]
  num_tokens: int = 0
  
class ICModel(nn.Module):
  def __init__(self, tokenizer, args: ICArgs = ICArgs()):
    super().__init__()
    self.tokenizer = tokenizer
    self.feature_extractor = utils.get_feature_extractor_for_model(args.visual_encoder, train=False)
    self.image_token = self.tokenizer.cls_token_id
    self.args = args
    self.num_tokens = args.num_tokens
    self.emb_dim = 256

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
    self.lm.config.pad_token_id = tokenizer.pad_token_id
    # self.token_idx = args.token_idx
    self.lm.resize_token_embeddings(len(tokenizer))

    self.input_embeddings = self.lm.get_input_embeddings()

    print("Restoring pretrained weights for the visual model.")
    if 'vit' in visual_encoder:
      # self.visual_model = ViTModel.from_pretrained(visual_encoder, ignore_mismatched_sizes=True)
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
    
    embedding_dim = self.input_embeddings.embedding_dim # * 32
    self.cap_hidden_fcs = nn.ModuleList([])
    self.vis_hidden_fcs = nn.ModuleList([])
    
    for layer_idx in self.args.text_emb_layers:
      if (layer_idx == -1 or layer_idx == self.lm.config.num_hidden_layers) and ('bert' not in opt_version):
        in_dim = 4096
        
        self.cap_hidden_fcs.append(
            layer.Adapter(in_dim, self.emb_dim, num_input_tokens=self.args.num_tokens,
                              num_output_tokens=1, mode = 'qformer'))
        self.vis_hidden_fcs.append(
            layer.Adapter(in_dim, self.emb_dim, num_input_tokens=self.args.num_tokens,
                              num_output_tokens=1, mode = 'qformer'))
      elif layer_idx < self.lm.config.num_hidden_layers:
        self.cap_hidden_fcs.append(
            layer.Adapter(self.lm.config.hidden_size, self.emb_dim, 
                              num_input_tokens=self.args.num_tokens, num_output_tokens=1, mode = 'qformer'))
        self.vis_hidden_fcs.append(
            layer.Adapter(self.lm.config.hidden_size, self.emb_dim, 
                              num_input_tokens=self.args.num_tokens, num_output_tokens=1, mode = 'qformer'))
      else:
        raise ValueError(f'Embedding of layer {layer_idx} was requested but model only has {self.lm.config.num_hidden_layers} layers.')
    self.visual_embeddings = nn.Linear(hidden_size, embedding_dim) # (768, 32 * 4096)
    # print("hidden_size: ", hidden_size)
    # print("embedding_dim: ", embedding_dim)
    # self.freeze_layer(self.visual_embeddings)
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
  def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

  def get_visual_embs(self, pixel_values: torch.FloatTensor):
    # Extract visual embeddings from the vision encoder.
    if 'vit' in self.visual_model_name:
      bs, _, _, _ = pixel_values.shape
      # print(pixel_values.shape)
      outputs = self.visual_model(pixel_values)
      # outputs = torch.stack([outputs.last_hidden_state[i, 167:168, :] for i in range(bs)], axis=0)
      outputs = outputs.last_hidden_state
      # print(outputs.shape)
      # outputs = outputs.pooler_output
      # visual adapter에 넣고 visual emb 뽑기
      # vis_input_embs = self.visual_input_embeddings(pixel_values) # qformer로 해야하나? 어떻게 할 수 있지?
      # vis_hidden_states = []
      # for idx, fc_layer in zip(self.args.text_emb_layers, visual_embeddings):
      #   vit_hidden_state = torch.stack([outputs.last_hidden_state[i, 167:168, :] for i in range(bs)], axis=0)
      #   vis_input_embedding = torch.stack([vis_input_embs[i, cap_embedding_idx[i]:cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
      #   vis_hidden_states.append(fc_layer(input_hidden_state, cap_input_embedding))  # (N, seq_len, 2048)
      visual_embs = self.visual_embeddings(outputs)
      # print(visual_embs.shape)
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 197, 4096))
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
    caption_len: Optional[torch.LongTensor] = None,
    neg_labels: Optional[torch.LongTensor] = None,
    neg_caption_len: Optional[torch.LongTensor] = None,
    input_prefix: Optional[str] = None
  ):
    visual_embs = self.get_visual_embs(pixel_values)

    batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens
    if labels is not None:
      assert labels.shape[0] == batch_size, (visual_embs.shape, labels.shape)
    if neg_labels is None:
      input_embs = self.input_embeddings(labels)  # (N, T, D)
      
      cap_embedding_idx = caption_len - 1
      # print(cap_embedding_idx)
      # if input_prefix is not None:
      #   prompt_ids = self.tokenizer(input_prefix, add_special_tokens=False, return_tensors="pt").input_ids
      #   prompt_ids = prompt_ids.to(visual_embs.device)
      #   prompt_embs = self.input_embeddings(prompt_ids)
      #   prompt_embs = prompt_embs.repeat(batch_size, 1, 1)
      #   print(f'Adding prefix "{input_prefix}" to retrieval.')
      #   # Add prompt embeddings.
      #   prefix_embs = prompt_embs
      #   input_embs = torch.cat([prefix_embs, input_embs], axis=1)
      #   cap_embedding_idx += prefix_embs.shape[1]
      #   labels = torch.cat([
      #     torch.zeros(prefix_embs.shape[:2], dtype=torch.int64).to(labels.device) - 100,
      #     labels
      #   ], axis=1)
      #   assert prompt_embs.shape[0] == batch_size, prompt_embs.shape
      #   assert prompt_embs.shape[2] == input_embs.shape[2], prompt_embs.shape
      #   assert len(prompt_embs.shape) == 3, prompt_embs.shape
      ### LLM에 넣기
      cap_output = self.lm(inputs_embeds=input_embs,
                            labels = labels,
                            output_hidden_states=True)

      visual_output = self.lm(inputs_embeds=visual_embs,
                              output_hidden_states=True)
      ### Adapter 생성
      cap_hidden_fcs = self.cap_hidden_fcs
      vis_hidden_fcs = self.vis_hidden_fcs
      
      ### output 생성
      cap_hidden_states = []
      for idx, fc_layer in zip(self.args.text_emb_layers, cap_hidden_fcs):
        input_hidden_state = torch.stack([cap_output.hidden_states[idx][i, cap_embedding_idx[i]:cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
        cap_input_embedding = torch.stack([input_embs[i, cap_embedding_idx[i]:cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
        cap_hidden_states.append(fc_layer(input_hidden_state, cap_input_embedding))  # (N, seq_len, 2048)
      
      visual_hidden_states = []
      for idx, fc_layer in zip(self.args.text_emb_layers, vis_hidden_fcs):
        input_hidden_state = torch.stack([visual_output.hidden_states[idx][i, 0:1, :] for i in range(batch_size)], axis=0)
        vis_input_embedding = torch.stack([visual_embs[i, 0:1, :] for i in range(batch_size)], axis=0)
        visual_hidden_states.append(fc_layer(input_hidden_state, vis_input_embedding))  # (N, seq_len, 2048)
      
      cap_last_hidden_states = torch.stack(cap_hidden_states, dim=-1).sum(dim=-1)
      visual_last_hidden_states = torch.stack(visual_hidden_states, dim=-1).sum(dim=-1)
      
      visual_embs = visual_last_hidden_states[:, 0, :]
      visual_embs = visual_embs / visual_embs.norm(dim=1, keepdim=True)
      cap_embs = cap_last_hidden_states[:, 0, :]
      cap_embs = cap_embs / cap_embs.norm(dim=1, keepdim=True)
      logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
      logit_scale = logit_scale.exp()
      visual_embs = logit_scale * visual_embs

      return cap_output, visual_output, cap_embs, visual_embs
    else:
      input_embs = self.input_embeddings(labels)  # (N, T, D)
      neg_input_embs = self.input_embeddings(neg_labels)  # (N, T, D)
      
      cap_embedding_idx = caption_len - 1
      neg_cap_embedding_idx = neg_caption_len - 1
      # print(cap_embedding_idx)
      ### LLM에 넣기
      cap_output = self.lm(inputs_embeds=input_embs,
                            labels = labels,
                            output_hidden_states=True)
      
      neg_cap_output = self.lm(inputs_embeds=neg_input_embs,
                            labels = neg_labels,
                            output_hidden_states=True)

      visual_output = self.lm(inputs_embeds=visual_embs,
                              output_hidden_states=True)
      ### Adapter 생성
      cap_hidden_fcs = self.cap_hidden_fcs
      vis_hidden_fcs = self.vis_hidden_fcs
      
      ### output 생성
      cap_hidden_states = []
      for idx, fc_layer in zip(self.args.text_emb_layers, cap_hidden_fcs):
        cap_input_hidden_state = torch.stack([cap_output.hidden_states[idx][i, cap_embedding_idx[i]:cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
        cap_input_embedding = torch.stack([input_embs[i, cap_embedding_idx[i]:cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
        cap_hidden_states.append(fc_layer(cap_input_hidden_state, cap_input_embedding))  # (N, seq_len, 2048)
      
      neg_cap_hidden_states = []
      for idx, fc_layer in zip(self.args.text_emb_layers, cap_hidden_fcs):
        neg_input_hidden_state = torch.stack([neg_cap_output.hidden_states[idx][i, neg_cap_embedding_idx[i]:neg_cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
        neg_cap_input_embedding = torch.stack([neg_input_embs[i, neg_cap_embedding_idx[i]:neg_cap_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
        neg_cap_hidden_states.append(fc_layer(neg_input_hidden_state, neg_cap_input_embedding))  # (N, seq_len, 2048)
      
      visual_hidden_states = []
      for idx, fc_layer in zip(self.args.text_emb_layers, vis_hidden_fcs):
        vis_input_hidden_state = torch.stack([visual_output.hidden_states[idx][i, 0:1, :] for i in range(batch_size)], axis=0)
        vis_input_embedding = torch.stack([visual_embs[i, 0:1, :] for i in range(batch_size)], axis=0)
        visual_hidden_states.append(fc_layer(vis_input_hidden_state, vis_input_embedding))  # (N, seq_len, 2048)
      
      # print("cap_hidden_states: ", cap_hidden_states)
      # print("neg_cap_hidden_states: ", neg_cap_hidden_states)
      
      cap_last_hidden_states = torch.stack(cap_hidden_states, dim=-1).sum(dim=-1)
      neg_cap_last_hidden_states = torch.stack(neg_cap_hidden_states, dim=-1).sum(dim=-1)
      visual_last_hidden_states = torch.stack(visual_hidden_states, dim=-1).sum(dim=-1)
      
      visual_embs = visual_last_hidden_states[:, 0, :]
      visual_embs = visual_embs / visual_embs.norm(dim=1, keepdim=True)
      cap_embs = cap_last_hidden_states[:, 0, :]
      cap_embs = cap_embs / cap_embs.norm(dim=1, keepdim=True)
      neg_cap_embs = neg_cap_last_hidden_states[:, 0, :]
      neg_cap_embs = neg_cap_embs / neg_cap_embs.norm(dim=1, keepdim=True)
      logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
      logit_scale = logit_scale.exp()
      visual_embs = logit_scale * visual_embs

      return cap_output, visual_output, cap_embs, neg_cap_embs, visual_embs

  
class IC(nn.Module):
  def __init__(self, tokenizer, model_args: Optional[ICArgs] = None):
    super().__init__()
    self.model = ICModel(tokenizer, model_args)


  def __call__(self, images: Tensor, tgt_tokens: Optional[Tensor] = None, 
               caption_len: Optional[Tensor] = None, neg_tgt_tokens: Optional[Tensor] = None, 
               neg_caption_len: Optional[Tensor] = None, input_prefix: Optional[str] = None) -> Tensor:
      output = self.model(
        pixel_values = images,
        labels = tgt_tokens,
        caption_len = caption_len,
        neg_labels = neg_tgt_tokens,
        neg_caption_len = neg_caption_len,
        input_prefix = input_prefix)
      return output