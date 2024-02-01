import cv2
import pywt
import numpy as np

def embed_secret_image(original_image_path, secret_image_path, output_image_path):
    # Orjinal resmi yükle
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # Gizli resmi yükle
    secret_image = cv2.imread(secret_image_path, cv2.IMREAD_GRAYSCALE)

    # Dosya yükleme başarısız olduysa hata mesajını yazdır ve çık
    if original_image is None or secret_image is None:
        print("Dosya yükleme hatası. Lütfen dosya yollarını kontrol edin.")
        return

    # Orjinal resmin boyutlarını kontrol et, tek sayı ise bir piksel ekleyerek çift sayıya tamamla
    if original_image.shape[0] % 2 == 1:
        original_image = cv2.copyMakeBorder(original_image, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if original_image.shape[1] % 2 == 1:
        original_image = cv2.copyMakeBorder(original_image, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=0)

    # Orjinal resmi DWT ile işle
    coeffs = pywt.dwt2(original_image, 'bior1.3')
    cA, (cH, cV, cD) = coeffs

    # cA katmanındaki katsayıları DCT ile işle
    cA_dct = cv2.dct(np.float32(cA))

    # Yüksek frekans bileşenlerini al
    high_freq_coefficients = cA_dct[secret_image.shape[0]:, secret_image.shape[1]:]

    # Gizli resmi yüksek frekans bileşenlerine göm
    high_freq_coefficients[:secret_image.shape[0], :secret_image.shape[1]] = secret_image.astype(np.float32)

    # Ters DCT işlemi
    cA_embedded = cv2.idct(cA_dct)

    # Diğer katsayıları ve gizli resmi birleştir
    coeffs_embedded = (cA_embedded, (cH, cV, cD))

    # Ters DWT işlemi
    embedded_image = pywt.idwt2(coeffs_embedded, 'bior1.3')

    # Gömülü resmi kaydet
    cv2.imwrite(output_image_path, embedded_image)

if __name__ == "__main__":
    # Orjinal resim, gizli resim ve çıktı resmi dosya yolları
    original_image_path = "DWT-DCT\\original_image.jpg"
    secret_image_path = "DWT-DCT\\secret_image.jpg"
    output_image_path = "DWT-DCT\\output_image_cA.jpg"

    # Gizli resmi göm
    embed_secret_image(original_image_path, secret_image_path, output_image_path)
