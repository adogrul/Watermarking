import numpy as np
import pywt
from PIL import Image

def embed_image(original_image_path, secret_image_path, output_image_path):
    # Ana resmi yükle ve gri tonlamalı hale getir
    original_image = Image.open(original_image_path).convert("L")
    original_array = np.array(original_image)

    # Gizli resmi yükle ve gri tonlamalı hale getir
    secret_image = Image.open(secret_image_path).convert("L")
    secret_array = np.array(secret_image)

    # Boyutları kontrol et
    if secret_array.shape[0] > original_array.shape[0] or secret_array.shape[1] > original_array.shape[1]:
        raise ValueError("Gizli resmin boyutları ana resmin boyutlarından büyük olamaz.")

    # Ana resmi Dalgalet Dönüşümü (DWT) uygula
    coeffs = pywt.dwt2(original_array, 'haar')

    # Katsayıları al
    cA, (cH, cV, cD) = coeffs

    # Gizli resmi genişlet
    target_height, target_width = cA.shape[0], cA.shape[1]
    secret_array_expanded = np.array(secret_image.resize((target_width, target_height)))

    # Boyutları kontrol et ve gizli resmi cA katmanına göm
    cV[:secret_array_expanded.shape[0], :secret_array_expanded.shape[1]] = secret_array_expanded

    # Yeni katsayıları oluştur
    new_coeffs = (cA, (cH, cV, cD))

    # Yeni resmi elde et
    embedded_image_array = pywt.idwt2(new_coeffs, 'haar')

    # Pikselleri 0 ile 255 arasına kırp
    embedded_image_array = np.clip(embedded_image_array, 0, 255)

    # Yeni resmi kaydet
    embedded_image = Image.fromarray(embedded_image_array.astype(np.uint8))
    embedded_image.save(output_image_path)

# Örnek kullanım
original_image_path = "original_image.jpg"
secret_image_path = "secret_image.jpg"
output_image_path = "output_image_cV.jpg"

embed_image(original_image_path, secret_image_path, output_image_path)
