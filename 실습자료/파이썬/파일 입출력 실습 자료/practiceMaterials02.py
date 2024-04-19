# XML 파일 만들기 실습
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

# 루트 엘리멘트 생성
root = ET.Element('bookstore')

book_info = {
    "books_title01" : "Harry Potter", "books_author01" : "J. K. Rowling", "books_year01" : "2005", "books_price01" : "29.99",
    "books_title02": "In to XML", "books_author02": "John Doe", "books_year02": "2010", "books_price02": "38.82",
}

# 첫 번째 책 엘리먼트 생성
book1 = ET.SubElement(root, "book")
book1.set("category", "fiction")

title1 = ET.SubElement(book1, "title")
title1.set("long", "en")
title1.text = book_info["books_title01"]

author1 = ET.SubElement(book1, "author")
author1.text = book_info["books_author01"]

year1 = ET.SubElement(book1, "year")
year1.text = book_info["books_year01"]

price1 = ET.SubElement(book1, "price")
price1.text = book_info["books_price01"]

# 두 번째 책 엘리먼트 생성
book2 = ET.SubElement(root, "book")
book2.set("category", "non-fiction")

title2 = ET.SubElement(book2, "title")
title2.set("long", "en")
title2.text = book_info["books_title02"]

author2 = ET.SubElement(book2, "author")
author2.text = book_info["books_author02"]

year2 = ET.SubElement(book2, "year")
year2.text = book_info["books_year02"]

price2 = ET.SubElement(book2, "price")
price2.text = book_info["books_price02"]

tree = ET.ElementTree(root)

# XML을 문자열로 변환하여 minidom에 로드
xml_str = ET.tostring(root, encoding='utf-8')
xml_dom = minidom.parseString(xml_str)

# 들여쓰기 적용
pretty_xml_str = xml_dom.toprettyxml(indent="  ")

# 파일에 저장
with open("books.xml", "w") as f:
    f.write(pretty_xml_str)

