import cv2
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sift_features.pkl")

sift = cv2.SIFT_create()
dataset_cache = []
cache_loaded = False

def load_dataset_to_memory():
    global dataset_cache, cache_loaded
    
    if cache_loaded:
        return dataset_cache
        
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_PATH}. Silakan jalankan scripts/build_features.py terlebih dahulu!")

    print("Memuat pre-computed SIFT features dari file...")
    
    with open(MODEL_PATH, 'rb') as f:
        raw_data = pickle.load(f)
        
    for data in raw_data:
        # Rakit kembali tuple menjadi objek cv2.KeyPoint
        reconstructed_kp = [
            cv2.KeyPoint(x=x, y=y, size=s, angle=a, response=r, octave=o, class_id=c)
            for x, y, s, a, r, o, c in data["kp_data"]
        ]
        
        dataset_cache.append({
            "category": data["category"],
            "item": data["item"],
            "image": data["image"],
            "kp": reconstructed_kp,
            "des": data["des"],
            "width": data["width"],
            "height": data["height"]
        })
        
    cache_loaded = True
    print(f"Selesai! Memuat {len(dataset_cache)} template objek secara instan.")
    return dataset_cache

def get_dataset_cache():
    global dataset_cache, cache_loaded
    if not cache_loaded:
        load_dataset_to_memory()
    return dataset_cache

def get_sift_detector():
    return sift