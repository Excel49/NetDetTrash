from flask import Blueprint, request, jsonify, render_template, Response, send_from_directory
import cv2
import numpy as np
import base64
import os
import json

# Import modul inti
from src.core.detector import detect_objects
from src.core.dataset import get_sift_detector
from src.core.preprocessor import preprocess_image, preprocess_image_steps
from src.utils.visualizer import (
    process_classification_result, 
    generate_frames, 
    encode_image_base64, 
    draw_sift_matches,
    draw_ransac_inliers_outliers
)

main_bp = Blueprint('main', __name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VISUAL_DIR = os.path.join(BASE_DIR, "models", "visualizations")

@main_bp.route('/')
def index():
    return render_template('index.html')

def _decode_image_from_base64_string(image_data):
    """Fungsi bantuan untuk membaca format gambar dari kamera (Base64)"""
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Format gambar tidak valid')
    return img

def _hitung_skor_matching(result):
    """Skor kualitas matching SIFT (bukan metrik evaluasi klasifikasi formal)."""
    confidence_match = max(0.0, min(0.999, result["confidence"] / 100))
    kp_dataset = max(1, len(result["kp_dataset"]))
    coverage_fitur = result["inliers"] / (result["inliers"] + max(1, kp_dataset - result["inliers"]) * 0.08)
    coverage_fitur = max(0.0, min(0.999, coverage_fitur))
    skor_kemiripan = 0 if confidence_match + coverage_fitur == 0 else (2 * confidence_match * coverage_fitur) / (confidence_match + coverage_fitur)
    return {
        "confidence_match": f"{confidence_match * 100:.1f}%",
        "coverage_fitur": f"{coverage_fitur * 100:.1f}%",
        "skor_kemiripan": f"{skor_kemiripan * 100:.1f}%"
    }

def _buat_payload_deteksi(img):
    """Response JSON untuk landing page."""
    results = detect_objects(img)
    hasil = process_classification_result(img, results)
    hasil["count"] = len(results)
    hasil["metrics_note"] = "Skor kualitas matching SIFT, bukan metrik evaluasi klasifikasi formal (precision/recall/accuracy)."

    for index, result in enumerate(results):
        if index < len(hasil["matches"]):
            hasil["matches"][index].update(_hitung_skor_matching(result))

    if results:
        confidence_list = [float(m["confidence_match"].replace("%", "")) for m in hasil["matches"]]
        coverage_list = [float(m["coverage_fitur"].replace("%", "")) for m in hasil["matches"]]
        kemiripan_list = [float(m["skor_kemiripan"].replace("%", "")) for m in hasil["matches"]]
        hasil["summary_metrics"] = {
            "confidence_match": f"{np.mean(confidence_list):.1f}%",
            "coverage_fitur": f"{np.mean(coverage_list):.1f}%",
            "skor_kemiripan": f"{np.mean(kemiripan_list):.1f}%"
        }
    else:
        hasil["summary_metrics"] = {"confidence_match": "0.0%", "coverage_fitur": "0.0%", "skor_kemiripan": "0.0%"}

    return hasil

def _render_analysis_page(img):
    """
    Fungsi krusial untuk membongkar step-by-step proses SIFT 
    dan mengirimkannya ke analysis.html
    """
    # Preprocessing steps: grayscale, CLAHE, blur
    gray, clahe, blur = preprocess_image_steps(img)
    
    # Visualisasi Tambahan: Difference of Gaussian (DoG) simulasi
    # SIFT mencari ekstremum di ruang DoG. Kita simulasikan 1 level DoG untuk visualisasi.
    blur1 = cv2.GaussianBlur(gray, (0, 0), 1.6)
    blur2 = cv2.GaussianBlur(gray, (0, 0), 1.6 * 1.414) # k = sqrt(2)
    dog = cv2.subtract(blur1, blur2)
    dog_vis = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    gray_b64 = encode_image_base64(gray)
    clahe_b64 = encode_image_base64(clahe)
    blur_b64 = encode_image_base64(blur)
    dog_b64 = encode_image_base64(cv2.applyColorMap(dog_vis, cv2.COLORMAP_JET)) # Warnai heatmap agar jelas

    # Hitung histogram intensitas untuk grafik perbandingan preprocessing
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
    hist_clahe = cv2.calcHist([clahe], [0], None, [256], [0, 256]).flatten().tolist()
    hist_blur = cv2.calcHist([blur], [0], None, [256], [0, 256]).flatten().tolist()

    # Statistik intensitas per tahap
    stats_preprocessing = {
        "gray":  {"mean": f"{gray.mean():.1f}",  "std": f"{gray.std():.1f}",  "min": int(gray.min()),  "max": int(gray.max())},
        "clahe": {"mean": f"{clahe.mean():.1f}", "std": f"{clahe.std():.1f}", "min": int(clahe.min()), "max": int(clahe.max())},
        "blur":  {"mean": f"{blur.mean():.1f}",  "std": f"{blur.std():.1f}",  "min": int(blur.min()),  "max": int(blur.max())}
    }

    # Keypoint detection pada hasil preprocessing akhir
    sift = get_sift_detector()
    kp, _ = sift.detectAndCompute(blur, None)
    kp_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_b64 = encode_image_base64(kp_img)
    
    # Keypoint detection pada citra asli (tanpa preprocessing CLAHE & Blur)
    kp_raw, _ = sift.detectAndCompute(gray, None)
    kp_raw_img = cv2.drawKeypoints(img, kp_raw, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_raw_b64 = encode_image_base64(kp_raw_img)
    
    # Deteksi objek menggunakan pipeline utama
    results = detect_objects(img)

    out_img = img.copy()
    analysis_details = []

    
    for r in results:
        # Gambar Bounding Box di gambar akhir
        box = np.int32(r["box"])
        color = (0, 255, 0) if r["category"] == 'organik' else (0, 165, 255)
        cv2.polylines(out_img, [box], True, color, 3, cv2.LINE_AA)
        
        # Buat gambar garis-garis pencocokan SIFT (Feature Matching) - Good Matches awal
        match_img = draw_sift_matches(r, img)
        match_b64 = encode_image_base64(match_img)
        
        # Buat gambar garis inliers vs outliers dari RANSAC
        ransac_img = draw_ransac_inliers_outliers(r, img)
        ransac_b64 = encode_image_base64(ransac_img)
        
        tpl_b64 = encode_image_base64(r['dataset_img'])
        
        analysis_details.append({
            'item': r['item'],
            'category': r['category'],
            'good_matches': r.get('good_matches_count', 0),
            'inliers': r['inliers'],
            'ransac_rate': f"{(r['inliers'] / max(1, r.get('good_matches_count', 1))) * 100:.1f}%",
            'confidence': f"{r['confidence']:.1f}%",
            **_hitung_skor_matching(r),
            'kp_template_cnt': len(r['kp_dataset']),
            'template_b64': tpl_b64,
            'match_b64': match_b64,
            'ransac_b64': ransac_b64
        })
        
    final_b64 = encode_image_base64(out_img)
    
    # Kirim semua gambar Base64 + data histogram ke frontend HTML
    return render_template('analysis.html', 
                           gray_b64=gray_b64,
                           clahe_b64=clahe_b64,
                           blur_b64=blur_b64,
                           dog_b64=dog_b64,
                           kp_b64=kp_b64,
                           kp_raw_b64=kp_raw_b64,
                           final_b64=final_b64,
                           total_kp=len(kp),
                           total_kp_raw=len(kp_raw),
                           details=analysis_details,
                           hist_gray=json.dumps(hist_gray),
                           hist_clahe=json.dumps(hist_clahe),
                           hist_blur=json.dumps(hist_blur),
                           stats_prep=stats_preprocessing)

# ==========================================
# ENDPOINT UNTUK WEB INTERFACE (HTML)
# ==========================================

@main_bp.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint untuk form Upload Gambar di index.html"""
    if 'file' not in request.files:
        return render_template('index.html', error='Tidak ada file yang diunggah')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='File kosong')
        
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return render_template('index.html', error='File tidak valid')
            
        return _render_analysis_page(img)
    except Exception as e:
        return render_template('index.html', error=str(e))

@main_bp.route('/analyze_base64_form', methods=['POST'])
def analyze_base64_form():
    """Endpoint untuk tombol Capture Kamera di index.html"""
    try:
        image_data = request.form.get('image')
        if not image_data:
            return render_template('index.html', error='Tidak ada data gambar')
            
        img = _decode_image_from_base64_string(image_data)
        return _render_analysis_page(img)
    except Exception as e:
        return render_template('index.html', error=str(e))

@main_bp.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Deteksi upload untuk landing page."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'File gambar tidak valid'}), 400
        return jsonify(_buat_payload_deteksi(img))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/analyze_base64', methods=['POST'])
def api_analyze_base64():
    """Deteksi capture untuk landing page."""
    try:
        image_data = request.form.get('image') or (request.json or {}).get('image')
        if not image_data:
            return jsonify({'error': 'Tidak ada data gambar'}), 400
        img = _decode_image_from_base64_string(image_data)
        return jsonify(_buat_payload_deteksi(img))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/dataset_keypoints')
def api_dataset_keypoints():
    """Daftar visualisasi keypoint dataset."""
    hasil = {"organik": {}, "anorganik": {}}
    for kategori in hasil.keys():
        kategori_dir = os.path.join(VISUAL_DIR, kategori)
        if not os.path.isdir(kategori_dir):
            continue
        for item in os.listdir(kategori_dir):
            item_dir = os.path.join(kategori_dir, item)
            if not os.path.isdir(item_dir):
                continue
            gambar = []
            for nama_file in os.listdir(item_dir):
                if nama_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    gambar.append({
                        "name": nama_file,
                        "url": f"/dataset_keypoints/{kategori}/{item}/{nama_file}"
                    })
            hasil[kategori][item] = gambar
    return jsonify(hasil)

@main_bp.route('/dataset_keypoints/<path:filename>')
def dataset_keypoints_file(filename):
    """Serve gambar keypoint dataset."""
    return send_from_directory(VISUAL_DIR, filename)

# ==========================================
# ENDPOINT UNTUK API / REALTIME CAMERA
# ==========================================

@main_bp.route('/video_feed')
def video_feed():
    """Endpoint untuk stream kamera realtime (MJPEG)"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
