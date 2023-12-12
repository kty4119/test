# import json

# json_path = '/home/kty4119/test/coco/annotations/captions_val2017.json'

# def extract_image_ids(data_list):
#     image_ids = [item['image_id'] for item in data_list]
#     return image_ids

# def format_to_12_digits(image_list):
#     jpg_list =[]
#     for number in image_list:
#         # 숫자를 문자열로 변환
#         str_number = str(number)
    
#         # 현재 문자열의 자릿수 확인
#         num_digits = len(str_number)
    
#         # 부족한 자릿수만큼 0을 앞에 추가하여 12자리 문자열 생성
#         formatted_number = '0' * (12 - num_digits) + str_number +'.jpg'
#         jpg_list.append(formatted_number)
    
#     return jpg_list

# with open(json_path, 'r') as file:
#     data = json.load(file)
#     # print(len(data))
#     print(data.keys())
#     print(data['images'][1])
#     image_list = extract_image_ids(data['annotations'])
    
#     jpg_list = format_to_12_digits(image_list)
#     print(jpg_list)

# import glob
# img_list = glob.glob('/home/kty4119/coco/train2014/*.jpg')
# for i in range(10):
#     print(img_list[i])

# import os

# # 디렉토리 경로 설정
# directory = '/home/kty4119/coco/val2014/'  # 실제 경로로 변경해주세요

# # 디렉토리 내의 파일 목록 가져오기
# file_list = os.listdir(directory)

# # for i in range(10):
# #     print(file_list[i])

# # 파일명 변경
# for filename in file_list:
#     if filename.startswith('COCO_val2014_'):
#         new_filename = filename.replace('COCO_val2014_', '')
#         old_path = os.path.join(directory, filename)
#         new_path = os.path.join(directory, new_filename)
#         os.rename(old_path, new_path)
#         print(f'Renamed: {filename} -> {new_filename}')
    
    