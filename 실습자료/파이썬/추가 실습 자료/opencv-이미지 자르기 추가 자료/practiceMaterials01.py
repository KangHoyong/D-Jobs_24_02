# 추가 자료 : 좌표를 이용한 크롭 하는 방법

import cv2

# 이미지 파일을 읽어옵니다.
image = cv2.imread('cat_image01.png')

# 이미지의 높이와 너비를 가져옵니다.
height, width = image.shape[:2]

# 자를 영역의 좌표를 설정합니다.
x = 250  # 시작 x 좌표
y = 50   # 시작 y 좌표
w = 720  # 너비
h = 720  # 높이

# 이미지를 자릅니다.
cropped_image = image[y:y+h, x:x+w]

# 자른 이미지를 보여줍니다.
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
