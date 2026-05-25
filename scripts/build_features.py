import cv2
import os
import pickle
import numpy as np

# Konfigurasi path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(BASE_DIR)

from src.core.preprocessor import preprocess_image

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sift_features.pkl")
VIS_DIR = os.path.join(MODEL_DIR, "visualizations")

def draw_and_save_visualization(img, kp, category, item_folder, filename):
    """
    Fungsi untuk menggambar keypoints dan data statistiknya, 
    kemudian menyimpannya sebagai file gambar baru.
    """
    # 1. Klasifikasikan KP berdasarkan kekuatan kontras (response)
    if len(kp) > 0:
        avg_response = np.mean([k.response for k in kp])
    else:
        avg_response = 0

    strong_kp = [k for k in kp if k.response >= avg_response] # Gradien Kuat (Sudut/Tepi)
    texture_kp = [k for k in kp if k.response < avg_response] # Tekstur Halus (Bintik/Pola)

    # 2. Gambar Keypoints menggunakan DRAW_RICH_KEYPOINTS (menampilkan ukuran dan orientasi)
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 3. Buat kanvas hitam tambahan di bagian bawah gambar untuk panel teks
    h, w = img_kp.shape[:2]
    panel_height = 100
    
    # Buat kanvas baru yang lebih tinggi
    canvas = np.zeros((h + panel_height, max(w, 600), 3), dtype=np.uint8)
    canvas[:h, :w] = img_kp

    # 4. Tulis statistik ke atas panel hitam
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Kategori : {category.upper()} - {item_folder}", (15, h + 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"Total KP : {len(kp)} titik", (15, h + 55), font, 0.6, (0, 255, 0), 1)
    
    # Kolom sebelah kanan untuk detail Tekstur vs Gradien
    col_2 = 300 if w < 600 else w // 2
    cv2.putText(canvas, f"Tepi/Sudut Tajam (High Gradient) : {len(strong_kp)} KP", (col_2, h + 25), font, 0.5, (0, 165, 255), 1)
    cv2.putText(canvas, f"Tekstur Halus/Pola (Subtle Texture) : {len(texture_kp)} KP", (col_2, h + 50), font, 0.5, (255, 255, 0), 1)
    cv2.putText(canvas, f"Avg Contrast Response             : {avg_response:.4f}", (col_2, h + 75), font, 0.5, (200, 200, 200), 1)

    # 5. Pastikan folder output tersedia
    output_dir = os.path.join(VIS_DIR, category, item_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 6. Simpan gambar
    save_path = os.path.join(output_dir, f"vis_{filename}")
    cv2.imwrite(save_path, canvas)


def build_offline_features():

    print("Memulai ekstraksi fitur SIFT dari dataset...\n")
    sift = cv2.SIFT_create()
    dataset_features = []

    # Hapus file model lama jika ada
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    # Hapus folder visualisasi lama jika ada
    if os.path.exists(VIS_DIR):
        import shutil
        shutil.rmtree(VIS_DIR)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for category in ["organik", "anorganik"]:
        category_path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(category_path):
            continue
            
        for item_folder in os.listdir(category_path):
            item_path = os.path.join(category_path, item_folder)
            if not os.path.isdir(item_path):
                continue
                
            for file in os.listdir(item_path):
                img_path = os.path.join(item_path, file)
                if not os.path.isfile(img_path):
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                gambar_proses = preprocess_image(img)
                kp, des = sift.detectAndCompute(gambar_proses, None)
                
                if kp is not None and des is not None and len(kp) > 20:
                    # Ambil 200 fitur terbaik
                    idx = np.argsort([-k.response for k in kp])[:1500]
                    kp = [kp[i] for i in idx]
                    des = des[idx]
                    
                    # Panggil fungsi visualizer kita di sini
                    draw_and_save_visualization(img, kp, category, item_folder, file)
                    
                    # Konversi cv2.KeyPoint ke tuple agar bisa disave (.pkl)
                    kp_serialized = [
                        (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) 
                        for k in kp
                    ]
                    
                    dataset_features.append({
                        "category": category,
                        "item": item_folder,
                        "image": img,
                        "kp_data": kp_serialized,
                        "des": des,
                        "width": img.shape[1],
                        "height": img.shape[0]
                    })
                    print(f"[OK] Terekstrak & Tervisualisasi: {category}/{item_folder} ({len(kp)} Keypoints)")

    # Simpan fitur matematika ke file .pkl untuk load backend
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(dataset_features, f)
        
    print(f"\nSelesai! Fitur model disimpan di: {MODEL_PATH}")
    print(f"Gambar visualisasi Keypoints disimpan di: {VIS_DIR}")

if __name__ == "__main__":
    build_offline_features()
