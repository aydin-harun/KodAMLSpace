import cv2
import numpy as np
import os
from skimage.measure import shannon_entropy
import datetime

def is_blank_page(image_path, fill_threshold=0.005, entropy_threshold=0.5, margin_ratio=0.03, debug=False):
    """
    Siyah-beyaz TIF görseli için boş sayfa tespiti.

    Args:
        image_path (str): Görsel dosya yolu.
        fill_threshold (float): Dolu piksellerin sayfa oranı eşiği. 0.005 = %0.5.
        entropy_threshold (float): Shannon entropi eşiği. Düşükse boş sayfa olabilir.
        margin_ratio (float): ROI (kenar boşlukları) oranı. 0.03 = %3 kenar bırakılır.
        debug (bool): Adım adım görselleri kaydet ve detay yaz.

    Returns:
        is_blank (bool), fill_ratio (float), entropy (float)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"'{image_path}' yüklenemedi.")

    # ROI kırp (kenarlardaki parazitleri engelle)
    h, w = img.shape
    top = int(h * margin_ratio)
    bottom = int(h * (1 - margin_ratio))
    left = int(w * margin_ratio)
    right = int(w * (1 - margin_ratio))
    roi = img[top:bottom, left:right]

    # Gürültü azalt (median blur)
    denoised = cv2.medianBlur(roi, 3)

    # Adaptive threshold (sabit eşik yerine)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 15)

    # Dolu piksellerin oranı (non-zero pixel count)
    nonzero_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    fill_ratio = nonzero_pixels / total_pixels

    # Entropi (tekdüze boş sayfa tespiti için)
    entropy = shannon_entropy(thresh)

    if debug:
        print(f"💡 Entropi: {entropy:.3f} | Doluluk: %{fill_ratio*100:.3f}")
        cv2.imwrite("debug_roi.png", roi)
        cv2.imwrite("debug_thresh.png", thresh)

    # Eşik değerlere göre boşluk tespiti
    is_blank = (fill_ratio < fill_threshold) and (entropy < entropy_threshold)
    return is_blank, fill_ratio, entropy
print( datetime.datetime.now())
for i in range(1,7):
    image_path = f"{i}.tif"
    blank, ratio, entropy = is_blank_page(image_path, debug=True)
    print(f"Sayfa durumu: {image_path} {'Boş' if blank else 'Dolu'}")
    print(f"➤ Doluluk Oranı: %{ratio*100:.3f}")
    print(f"➤ Entropi: {entropy:.3f}")
print( datetime.datetime.now())
