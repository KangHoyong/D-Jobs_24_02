# 크롤링 실습 코드
# 셀레늄 이용한 Data Crawling 실습
import os
import time
import urllib.request
import socket

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
qurey = "night street"
service = Service('./chromedriver.exe')
driver = webdriver.Chrome(service=service)
driver.get("https://www.google.co.kr/imghp?h1=ko&tab=wi&authuser=0&ogb1") # 검색 사이트 주소 변경으로 인한 문제 발생

xpath = "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea" # 변경 되겠네

# 이미지 검색을 하기 위한 입력값 전달
keyword = driver.find_element(by=By.XPATH, value=xpath)
keyword.send_keys(qurey)

# 검색 버튼 클릭
button_xpath = "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button"
button = driver.find_element(by=By.XPATH, value=button_xpath)
button.click()

# 검색 스크롤 내리기
print(f"{qurey} 스크롤 내리는 중.....")
scroll_tag_name = 'body'
elem = driver.find_element(by=By.TAG_NAME, value=scroll_tag_name)

# 스크롤 내리다 보면 결과 더보기 버튼에 의해서 스크롤이 더이상 내려 가지 않는 문제가 발생
# 이러한 스크롤 내려 가도록 수정 필요
for i in range(60) :
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)

try :
    view_more_results_button_xpath = "/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[2]/div[2]/input"
    driver.find_element(by=By.XPATH, value=view_more_results_button_xpath).click()
    for i in range(60) :
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)
except :
    pass

save_directory = "./downloaded_images"
os.makedirs(save_directory, exist_ok=True)
images = driver.find_elements(By.CLASS_NAME, 'YQ4gaf')

# 썸네일 이미지가 아닌 원본 이미지 정보를 가져와서 저장
count = 0  # 다운로드 성공한 이미지의 개수를 세는 변수

for image in images:
    im_class = image.get_attribute("className")
    if im_class != "YQ4gaf":
        continue

    ActionChains(driver).move_to_element(image).click().perform()
    time.sleep(2)

    original_image_container_selector = ".sFlh5c.pT0Scc.iPVvYb"
    original_images = driver.find_elements(By.CSS_SELECTOR, original_image_container_selector)

    for org_img in original_images:
        org_image_url = org_img.get_attribute('src')
        org_image_url_data_src = org_img.get_attribute('data-src')
        if org_image_url:
            download_url = org_image_url
        elif org_image_url_data_src:
            download_url = org_image_url_data_src
        else:
            continue
        try:
            image_name = f"{qurey}_img_{count + 1}_org.png"
            image_path = os.path.join(save_directory, image_name)
            # 타임아웃 설정
            socket.setdefaulttimeout(10)  # 10초로 타임아웃 설정

            urllib.request.urlretrieve(download_url, image_path)
            time.sleep(2)
            print(f"Original image {count + 1} saved successfully : {image_path}")
            count += 1  # 다운로드 성공 시에만 count 증가
        except Exception as e:
            pass
            print(f"Failed to save image {count + 1}: {str(e)}")

driver.implicitly_wait(3)