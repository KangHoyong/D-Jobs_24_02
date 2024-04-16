# Quiz4
# while 문을 사용하여 1 ~ 1000 까지의 자연수 중 3의 배수의 합을 구하기

total = 0
num = 1

while num < 1001 :
    if num % 3 == 0 :
        total += num

    num +=1

print("1부터 1000 까지의 자연수 중 3의 배수의 합 : ", total)