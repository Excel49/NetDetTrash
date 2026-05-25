import cv2
import numpy as np

def preprocess_image(gambar):
    """CLAHE lalu Gaussian blur ringan."""
    abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gambar_clahe = clahe.apply(abu_abu)
    gambar_blur = cv2.GaussianBlur(gambar_clahe, (3, 3), 0)
    return gambar_blur

