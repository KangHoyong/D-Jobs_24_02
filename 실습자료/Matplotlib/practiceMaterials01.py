import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x)

# 2x2 그리드에 4개의 서브플롯 생성
plt.subplot(2,2,1) # 첫 번째 서브플롯
plt.plot(x, y1)
plt.title('Sin')

plt.subplot(2,2,2) # 두 번째 서브플롯
plt.plot(x, y2)
plt.title('Cos')

plt.subplot(2,2,3) # 세 번째 서브플롯
plt.plot(x, y3)
plt.title('Tan')

plt.subplot(2,2,4) # 네 번째 서브플롯
plt.plot(x, y4)
plt.title('Exp')

plt.tight_layout() # 그래프 간격 조정
plt.show()

