# 해당 코드는 이미지 폴더에서 이미지를 읽고 해당 이미지 비율에 맞게 리사이즈 작업 진행 후 이미지 저장
# 해당 코드를 실행하기 위해서는 다음과 같은 작업이 필요
"""
1. image_paths = glob.glob(r"./frame_image_save/*.jpg") -> 해당 경로에는 이미지가 있는 폴더 경로 지정
주의 --> *.jpg 로 되어있음 해당 타입은 폴더에 있는 이미지 타입에 맞게 지정 png 파일이면 *.png 변경
"""

import os
import glob
from PIL import Image

def expend2square(pil_img, background_color) :
    width, height = pil_img.size

    if width == height :
        return pil_img
    elif width > height :
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result

    else :
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def resize_with_padding(pil_img, new_size, background_color) :
    image = expend2square(pil_img, background_color)
    image = image.resize((new_size[0], new_size[1]),)
    return image

if __name__ == "__main__" :

    os.makedirs("./image_data", exist_ok=True)
    image_paths = glob.glob(r"./frame_image_save/*.jpg")

    for image_path in image_paths :
        image = Image.open(image_path)
        base_name = os.path.basename(image_path)

        image_new = resize_with_padding(image, (225, 255), (0, 0, 0))
        image_new.save(f"./image_data/{base_name}")

