import os
import cv2
import numpy as np
from xml.etree.ElementTree import parse
from polygon_utils import rle2_mask, mask2polygon

def xml_read(xml_path):

    root = parse(xml_path).getroot()
    images_info = root.findall('image')

    for image_info in images_info:
        image_name = image_info.attrib['name']
        image_path = os.path.join("./xml_image", image_name)

        image = cv2.imread(image_path)

        masks_info = image_info.findall('mask')
        for mask_info in masks_info:
            mask_rle = mask_info.attrib['rle']

            left = int(mask_info.attrib['left'])
            top = int(mask_info.attrib['top'])

            # 마스크에서 가로 세로 뽑느것
            mask_width = int(mask_info.attrib['width'])
            mask_height = int(mask_info.attrib['height'])

            # 마스크 크기의 빈 array
            mask_image = np.zeros((mask_height, mask_width), dtype=np.uint8)

            # rle를 보고 segment가 있는 부분은 1 아닌부분은 0인 리스트
            mask = rle2_mask(mask_width, mask_height, mask_rle)

            # mask에 255를 곱해서 색 칠할 부분 나눔
            mask *= 255
            # 색칠해줌
            mask_image[:mask_height, :mask_width] = mask

            # find contour
            polygon_list = mask2polygon(mask_image)
            print(polygon_list)

            # polygon
            append_polygon_list = []
            # polygon 리스트에 좌표값 더해주고 1차원 리스트 append# 폴리건 그리기
            for p in polygon_list:
               p = np.array(p).reshape((-1, 2)) + [left, top] # image scale 처리
               append_polygon_list.append(p.reshape(-1,).tolist()) # mask rle -> 디코딩 해서 처리 해야함

            # 바운딩 박스, 폴리곤 좌표 시각화
            for polygons in append_polygon_list :
                print("폴리곤 좌표 ", polygons)
                polygon_image = draw_polygon_with_bounding_box(image, polygons)

    cv2.imshow("test", polygon_image)
    cv2.waitKey(0)

def draw_polygon_with_bounding_box(image, polygon):

    # polygon_coords: 다각형의 꼭짓점 좌표가 포함된 리스트입니다. 홀수 인덱스는 x 좌표이고 짝수 인덱스는 y 좌표입니다.
    polygons = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]

    points = np.array(polygon, np.int32)
    points = points.reshape((-1, 1, 2))

    # Draw polylines
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # polygons 값에서 min max
    # min 함수를 사용하여 주어진 다각형 좌표 리스트 polygons에서 x 좌표가 가장 작은 값을 찾습니다
    # max 함수를 사용하여 주어진 다각형 좌표 리스트 polygons에서 y 좌표가 가장 높은 값을 찾습니다.
    min_x = min(polygons, key=lambda p: p[0])[0]
    max_x = max(polygons, key=lambda p: p[0])[0]
    min_y = min(polygons, key=lambda p: p[1])[1]
    max_y = max(polygons, key=lambda p: p[1])[1]

    # Draw bounding box
    cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color=(255, 0, 0), thickness=1)

    return image

if __name__ == "__main__":
    xml_read("./xml_annotation/k_fashion_cvat.xml")