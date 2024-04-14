# Quiz01.py

# 네 과목의 점수를 입력 받는다.
a,b,c,d = map(int, input().split())

if 0 <= a <= 100 and 0 <= b <= 100 and 0 <= c :
    if (a+b+c+d)/4 >= 80.0 :
        print("합격")

    else :
        print("불합격")

else :
    print("잘못된 점수 : 정수값을 입력")