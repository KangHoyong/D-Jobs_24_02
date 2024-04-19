import os

file_path = 'aaa.txt'

# 텍스트 파일이 없으면 생성
if not os.path.exists(file_path):
    with open(file_path, 'w') as file:
        print("Created aaa.txt")

while True:
    user_input = input("Enter content to append (Type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting....")
        break

    with open(file_path, 'a') as file:
        file.write(user_input + '\n')

    print("Content added successfully!\n")

print("Final content :")
with open(file_path, 'r') as file:
    file_data = file.read()
    print(file_data)


