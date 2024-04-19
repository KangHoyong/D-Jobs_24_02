# 다양한 정렬 알고리즘 소개 : 합병 정렬 (Merage Sort)

def merge_sort(arr) :
    if len(arr) > 1 :
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        # 재귀적으로 리스트 반으로 나누고 정렬
        merge_sort(left_half)
        merge_sort(right_half)

        # 두개의 정렬된 리스트를 병합
        i = j = k = 0
        while i < len(left_half) and j < len(right_half) :
            if left_half[i] < right_half[j] :
                arr[k] = left_half[i]
                i += 1

            else :
                arr[k] = right_half[j]
                j += 1

            k += 1

        while i < len(left_half) :
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half) :
            arr[k] = right_half[j]
            j += 1
            k += 1

# 예시
my_list = [12, 11 ,13 , 5, 6 , 7]
merge_sort(my_list)
print(my_list)