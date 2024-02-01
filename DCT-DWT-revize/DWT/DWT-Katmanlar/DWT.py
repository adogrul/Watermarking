import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt

# Resmi yükle
img = cv2.imread("baboon.jpeg", cv2.IMREAD_GRAYSCALE)

# DWT işlemi uygula
coeffs = pywt.dwt2(img, 'bior1.3')  # Bior1.3 dalgası kullanılabilir

# Düşük frekanslı ve yüksek frekanslı katsayıları al
cA, (cH, cV, cD) = coeffs

# DWT katsayılarını kaydet
cv2.imwrite("cA_coefficient.jpeg", cA)
cv2.imwrite("cH_coefficient.jpeg", cH)
cv2.imwrite("cV_coefficient.jpeg", cV)
cv2.imwrite("cD_coefficient.jpeg", cD)

# Orijinal görüntüyü kaydet
cv2.imwrite("original_image.jpeg", img)

# Görüntülerinizi görselleştirme
titles = ['Original Image', 'Approximation (cA)', 'Horizontal (cH)', 'Vertical (cV)', 'Diagonal (cD)']
images = [img, cA, cH, cV, cD]

for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.savefig("dwt-result-baboon.jpeg")
plt.show()
