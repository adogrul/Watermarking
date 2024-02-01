import cv2
import numpy as np

def resize_images(img1, img2):
    # İki resmin boyutlarını kontrol et
    if img1.shape != img2.shape:
        # Eğer boyutlar farklıysa, küçük olan resmi büyük olanın boyutuna getir
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return img1, img2

def calculate_psnr(img1, img2):
    # Resimleri yükle
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # Resimleri boyutlarına getir
    img1, img2 = resize_images(img1, img2)

    # PSNR hesapla
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Eğer MSE sıfırsa, PSNR sonsuza yaklaşır.
    
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr

# İki resim yolu
image1_path = "DWT-DCT\\output_image_cD.jpg"
image2_path = "DWT-DCT\\original_image.jpg"

try:
    # PSNR'yi hesapla
    psnr_value = calculate_psnr(image1_path, image2_path)
    # Sonucu yazdır
    print(f"\n\n\nPSNR değeri: {psnr_value} dB")

except ValueError as e:
    print(f"Hata: {e}")
