from transformers import AutoImageProcessor, ViTModel, AutoTokenizer, OPTModel, OPTForCausalLM, AutoModelForCausalLM
import json
from PIL import Image

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

img_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
hidden_size = img_model.config.hidden_size
visual_input_embeddings = img_model.get_input_embeddings()
img_token = image_processor(image, return_tensors="pt")
img_token = img_token.pixel_values
vis_input_embs = visual_input_embeddings(img_token)
vit_output = img_model(img_token)
pooler_output = vit_output.pooler_output
print(hidden_size)
print("vis_input_embs: ", vis_input_embs, vis_input_embs.shape)
print("pooler output: ", pooler_output, pooler_output.shape)

# text_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b")
# input_embeddings = text_model.get_input_embeddings()
# embedding_dim = input_embeddings.embedding_dim
# print(embedding_dim)

# in_dim = text_model.config.word_embed_proj_dim
# print(in_dim)