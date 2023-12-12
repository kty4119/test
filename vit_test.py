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

from transformers import AutoImageProcessor, ViTModel, AutoTokenizer, OPTModel, OPTForCausalLM, AutoModelForCausalLM
import torch
import torch.nn as nn
from PIL import Image, ImageFont
from datasets import load_dataset
import json

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
img_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") ## cls 토큰 추가

input_embeddings = img_model.get_input_embeddings()
img_token = image_processor(image, return_tensors="pt")
img_token = img_token.pixel_values
img_emb = input_embeddings(img_token)
original_size = 196 * 768
target_size = 17 * 4096
linear_layer = nn.Linear(original_size, target_size)
img_emb = linear_layer(img_emb.reshape(1, -1))
img_emb = img_emb.view(1, 17, 4096)
print(img_emb.shape)

### caption 임베딩

# text_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b")
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

tokenizer = AutoTokenizer.from_pretrained("/home/shared/hub/models--ty--alpaca-7b-wdiff")
text_model = AutoModelForCausalLM.from_pretrained("/home/shared/hub/models--ty--alpaca-7b-wdiff")
# # text_model.resize_token_embeddings(len(tokenizer))
input_embeddings = text_model.get_input_embeddings()
cap_token = tokenizer(caption, return_tensors="pt")
cap_token = cap_token.input_ids[0]
# print(cap_token)
cap_emb = input_embeddings(cap_token)

cap_emb = cap_emb.unsqueeze(0)
print(cap_emb.shape)
### image, caption을 LLM에 넣고 output뽑기
cap_outputs= text_model(inputs_embeds=cap_emb,
                       output_hidden_states=True)

img_outputs = text_model(inputs_embeds=img_emb,
                       output_hidden_states=True)
# print(cap_outputs.loss)
# print(img_outputs.loss)
# print(cap_outputs.logits.shape)
# print(img_outputs.logits.shape)
print(cap_outputs.hidden_states[32].shape)
print(img_outputs.hidden_states[32].shape)

original_size = 17 * 4096
target_size = 256
linear_layer = nn.Linear(original_size, target_size)

print(torch.stack(cap_outputs.hidden_states).reshape(1, -1).shape)

cap_outputs = linear_layer(cap_outputs.hidden_states[32].reshape(1, -1))
img_outputs = linear_layer(img_outputs.hidden_states[32].reshape(1, -1))

print(cap_outputs.shape, cap_outputs)
print(img_outputs.shape, img_outputs)