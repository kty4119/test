import glob
from PIL import Image
img_path = '/home/kty4119/coco/train2014/*.jpg'
img = glob.glob(img_path)
print(img[1])
image = Image.open(img[0])
image.show()