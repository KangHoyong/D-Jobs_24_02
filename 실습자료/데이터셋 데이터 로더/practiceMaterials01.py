# PyTorch Dataset 및 데이터 로도 속도 개선 방법 실험
import time
import os
import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def is_grayscale(img) :
    return img.mode == 'L'

class CustomDatasetNoCache(Dataset) :
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(os.path.join(image_paths, "*", "*.jpg"))
        self.transform = transform
        self.label_dict = {"dew": 0, "fogsmog": 1, "frost": 2, "glaze": 3, "hail": 4,
                           "lightning": 5, "rain": 6, "rainbow": 7, "rime": 8, "sandstorm": 9,
                           "snow": 10}
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if not is_grayscale(image):
            # 리눅스, 맥 기준
            # folder_name = image_path.split("/")
            # folder_name = folder_name[2]
            # 윈도우 기준 
            folder_name = image_path.split("\\")
            folder_name = folder_name[1]
            label = self.label_dict[folder_name]
        else:
            print("흑백 이미지 >>", image_path)
            return None, None
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

class CustomDatasetCache(Dataset) :

    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(os.path.join(image_paths, "*", "*.jpg"))
        self.transform = transform
        self.label_dict = {"dew": 0, "fogsmog": 1, "frost": 2, "glaze": 3, "hail": 4,
                           "lightning": 5, "rain": 6, "rainbow": 7, "rime": 8, "sandstorm": 9,
                           "snow": 10}
        self.cache = {} # 이미지 캐시

    def __getitem__(self, index) :
        if index in self.cache :
            image, label = self.cache[index]
        else :
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert("RGB")

            if not is_grayscale(image):
                # 리눅스, 맥 기준
                # folder_name = image_path.split("/")
                # folder_name = folder_name[2]
                # 윈도우 기준 
                folder_name = image_path.split("\\")
                folder_name = folder_name[1]
                label = self.label_dict[folder_name]
                self.cache[index] = (image, label)
            else :
                print("흑백 이미지 >>", image_path)
                return None, None

        if self.transform :
            image = self.transform(image)

        return image, label

    def __len__(self) :
        return len(self.image_paths)


transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 캐시 처리
dataset_with_cache = CustomDatasetCache(image_paths="./dataset", transform=transforms)
dataloader_with_cache = DataLoader(dataset_with_cache, batch_size=64, shuffle=True)

# 캐시 처리 No
dataset_with_no_cache = CustomDatasetNoCache(image_paths="./dataset", transform=transforms)
dataloader_with_no_cache = DataLoader(dataset_with_no_cache, batch_size=64, shuffle=True)

# 속도 비교
if __name__ == "__main__":

    transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # 캐시 처리
    dataset_with_cache = CustomDatasetCache(image_paths="./dataset", transform=transforms)
    dataloader_with_cache = DataLoader(dataset_with_cache, batch_size=64, shuffle=True)

    # 캐시 처리 No
    dataset_with_no_cache = CustomDatasetNoCache(image_paths="./dataset", transform=transforms)
    dataloader_with_no_cache = DataLoader(dataset_with_no_cache, batch_size=64, shuffle=True)

    # 속도 비교 
    start_time_1 = time.time()
    for image, label in dataloader_with_cache :
        pass
    end_time_1 = time.time()
    print("with cache - Elapsed Time : ", end_time_1 - start_time_1)
    start_time_2 = time.time()
    for image, label in dataloader_with_no_cache :
        pass
    end_time_2 = time.time()
    print("with No cache - Elapsed Time : ", end_time_2 - start_time_2)
