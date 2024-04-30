import numpy as np
import matplotlib.pyplot as plt
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
    image = Image.open("./apple01.jpeg")
    image_new = resize_with_padding(image, (225,255), (0,0,0))

    plt.imshow(image_new)
    plt.show()
