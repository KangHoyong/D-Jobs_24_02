# xml 파일 정보 가져오기

import xml.etree.ElementTree as ET

# XML 파일을 파싱하여 ElementTree 객체 생성
tree = ET.parse('books.xml')

# 최상위 엘리멘트 가져오기
root = tree.getroot()

# 각 책 정보 출력하기
for book in root.findall('book') :
     category = book.get('category')
     title = book.find('title').text
     author = book.find("author").text
     year = book.find("year").text
     price = book.find("price").text

     print(f"Category : {category}")
     print(f"Title : {title}")
     print(f"Author : {author}")
     print(f"Year : {year}")
     print(f"Price : {price}")
     print("***************************")