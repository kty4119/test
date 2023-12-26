# from transformers import ViTImageProcessor, ViTForImageClassification
# from PIL import Image
# import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

# from transformers import AutoImageProcessor, ViTModel, ViTFeatureExtractor, AutoTokenizer, OPTModel, OPTForCausalLM, AutoModelForCausalLM
from transformers import AutoImageProcessor, ViTFeatureExtractor, AutoTokenizer, OPTModel, OPTForCausalLM, AutoModelForCausalLM
import torch
import torch.nn as nn
from PIL import Image, ImageFont
from datasets import load_dataset
import json
import numpy as np

from ic import layer
from ic.custom_vit import ViTModel

### caption, image 불러오기
json_path = '/home/kty4119/coco/annotations/captions_train2014.json'
with open(json_path, 'r') as file:
    data = json.load(file)
    image_id = data['annotations'][0]['image_id']
    # 숫자를 문자열로 변환
    str_number = str(image_id)

    # 현재 문자열의 자릿수 확인
    num_digits = len(str_number)

    # 부족한 자릿수만큼 0을 앞에 추가하여 12자리 문자열 생성
    image_id = '0' * (12 - num_digits) + str_number +'.jpg'
    image = Image.open('/home/kty4119/coco/train2014/'+image_id)
    caption = data['annotations'][2]['caption']

### image 임베딩

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
img_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", ignore_mismatched_sizes=True) ## cls 토큰 추가

img_token = image_processor(image, return_tensors="pt")
img_token = img_token.pixel_values
# img_emb = input_embeddings(img_token)
output = img_model(img_token)
# output = output.pooler_output
# output = output.last_hidden_state
output = torch.stack([output.last_hidden_state[i, 167:168, :] for i in range(1)], axis=0)
print(output.shape)
# original_size = 198 * 768
original_size = 1 * 768
target_size = 1 * 4096
linear_layer = nn.Linear(original_size, target_size)
img_emb = linear_layer(output.reshape(1, -1))
img_emb = img_emb.view(1, 1, 4096)

### 이미지 LLM 모델 세팅
img_tokenizer = AutoTokenizer.from_pretrained("/home/shared/hub/models--ty--alpaca-7b-wdiff")
img_text_model = AutoModelForCausalLM.from_pretrained("/home/shared/hub/models--ty--alpaca-7b-wdiff")

special_tokens_dict = {"cls_token": "<|IMAGE summary|>"}
num_added_toks = img_tokenizer.add_special_tokens(special_tokens_dict)
img_text_model.resize_token_embeddings(len(img_tokenizer))
print("We have added", num_added_toks, "tokens")
token_idx = img_tokenizer("<|summary|>", add_special_tokens=True).input_ids
print("token_idx: ", token_idx)
cls_token_index = img_tokenizer.convert_tokens_to_ids([img_tokenizer.cls_token])[0]
print(cls_token_index)

### 텍스트 LLM 모델 세팅
tokenizer = AutoTokenizer.from_pretrained("/home/shared/hub/models--ty--alpaca-7b-wdiff")
text_model = AutoModelForCausalLM.from_pretrained("/home/shared/hub/models--ty--alpaca-7b-wdiff")

special_tokens_dict = {"cls_token": "<|CAP summary|>"}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
text_model.resize_token_embeddings(len(tokenizer))
print("We have added", num_added_toks, "tokens")
token_idx = tokenizer("<|summary|>", add_special_tokens=True).input_ids
print("token_idx: ", token_idx)
# tokenizer_idx = tokenizer.get_special_tokens_mask("<|summary|>")
# print(tokenizer_idx)
cls_token_index = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]
print(cls_token_index)

# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.

### caption 임베딩
input_embeddings = text_model.get_input_embeddings()
cap_tokenized_data = tokenizer(caption, return_tensors="pt")
cap_token = cap_tokenized_data.input_ids[0]
cap_len = cap_tokenized_data.attention_mask[0].sum()
print(cap_len)
# print(cap_token)
cap_emb = input_embeddings(cap_token)

cap_emb = cap_emb.unsqueeze(0)
print("cap_emb: ", cap_emb.shape)
### image, caption을 LLM에 넣고 output뽑기
cap_outputs= text_model(inputs_embeds=cap_emb,
                       output_hidden_states=True)

img_outputs = img_text_model(inputs_embeds=img_emb,
                       output_hidden_states=True)
last_hidden_states = cap_outputs.hidden_states[-1]
print("last_hidden_states: ", last_hidden_states.shape)
print("cls info: ", last_hidden_states[:, -1, :], last_hidden_states[:, -1, :].shape)
print(cap_outputs.logits[:,:,cls_token_index:cls_token_index+1])
# print(cap_outputs.loss)
# cls_token_embedding = last_hidden_states[:, cls_token_index, :]
# print("CLS Token Embedding:", cls_token_embedding)
# print(cap_outputs.loss)
# print(img_outputs.loss)
# print(cap_outputs.logits.shape)
# print(img_outputs.logits.shape)
print(cap_outputs.hidden_states[-1][:,0:1,:].shape)
print(img_outputs.hidden_states[-1][:,0:1,:].shape)
text_emb_layers = [-1]
cap_llm_hidden_states = []
cap_hidden_states = []
visual_llm_hidden_states = []
visual_hidden_states = []
original_size = 4096
target_size = 256
hidden_fcs = nn.ModuleList([])
hidden_fcs.append(layer.Adapter(4096, 256, num_input_tokens=1,
                              num_output_tokens=1))
# linear_layer = nn.Linear(original_size, target_size)

num_tokens = 8
cap_embedding_idx = cap_len - 1
for idx, fc_layer in zip(text_emb_layers, hidden_fcs):
    cap_input_hidden_state = torch.stack([cap_outputs.hidden_states[idx][i, 16:17, :] for i in range(1)], axis=0)
    print("cap_input_hidden_state: ", cap_input_hidden_state.shape)
    print(cap_input_hidden_state)
    cap_llm_hidden_states.append(cap_input_hidden_state)
    cap_hidden_states.append(fc_layer(cap_input_hidden_state))  # (N, seq_len, 2048)

    visual_input_hidden_state = torch.stack([img_outputs.hidden_states[idx][i, 16:17, :] for i in range(1)], axis=0)
    print("visual_input_hidden_state: ", visual_input_hidden_state.shape)
    visual_llm_hidden_states.append(visual_input_hidden_state)
    visual_hidden_states.append(fc_layer(visual_input_hidden_state))

cap_embedding = torch.stack(cap_hidden_states, dim=-1).sum(dim=-1)
visual_embedding = torch.stack(visual_hidden_states, dim=-1).sum(dim=-1)

visual_embs = visual_embedding[:, 0, :]
visual_embs = visual_embs / visual_embs.norm(dim=1, keepdim=True)
last_embedding = cap_embedding[:, 0, :]
last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
logit_scale = logit_scale.exp()
visual_embs = logit_scale * visual_embs

print(visual_embs.shape, visual_embs)
print(last_embedding.shape, last_embedding)

# print(torch.stack(cap_outputs.hidden_states).reshape(1, -1).shape)

# cap_outputs = linear_layer(cap_outputs.hidden_states[32].reshape(1, -1))
# img_outputs = linear_layer(img_outputs.hidden_states[32].reshape(1, -1))

# print(cap_outputs.shape, cap_outputs)
# print(img_outputs.shape, img_outputs)