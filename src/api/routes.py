from flask import Blueprint, request, jsonify, render_template, Response, send_from_directory
import cv2
import numpy as np
import base64
import os

# Import modul inti
from src.core.detector import detect_objects
from src.core.dataset import get_sift_detector
from src.core.preprocessor import preprocess_image
from src.utils.visualizer import (
    process_classification_result, 
    generate_frames, 
    encode_image_base64, 
    draw_sift_matches
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

def _hitung_metric_sift(result):
    """Metrik estimasi dari kualitas matching."""
    precision = max(0.0, min(0.999, result["confidence"] / 100))
    kp_dataset = max(1, len(result["kp_dataset"]))
    recall = result["inliers"] / (result["inliers"] + max(1, kp_dataset - result["inliers"]) * 0.08)
    recall = max(0.0, min(0.999, recall))
    f1 = 0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "precision": f"{precision * 100:.1f}%",
        "recall": f"{recall * 100:.1f}%",
        "f1_score": f"{f1 * 100:.1f}%"
    }

def _buat_payload_deteksi(img):
    """Response JSON untuk landing page."""
    results = detect_objects(img)
    hasil = process_classification_result(img, results)
    hasil["count"] = len(results)
    hasil["metrics_note"] = "Estimasi berbasis matching SIFT, bukan evaluasi ground-truth manual."

    for index, result in enumerate(results):
        if index < len(hasil["matches"]):
            hasil["matches"][index].update(_hitung_metric_sift(result))

    if results:
        precision_list = [float(m["precision"].replace("%", "")) for m in hasil["matches"]]
        recall_list = [float(m["recall"].replace("%", "")) for m in hasil["matches"]]
        f1_list = [float(m["f1_score"].replace("%", "")) for m in hasil["matches"]]
        hasil["summary_metrics"] = {
            "precision": f"{np.mean(precision_list):.1f}%",
            "recall": f"{np.mean(recall_list):.1f}%",
            "f1_score": f"{np.mean(f1_list):.1f}%"
        }
    else:
        hasil["summary_metrics"] = {"precision": "0.0%", "recall": "0.0%", "f1_score": "0.0%"}

    return hasil

def _render_analysis_page(img):
    """
    Fungsi krusial untuk membongkar step-by-step proses SIFT 
    dan mengirimkannya ke analysis.html
    """
    # Step 1: Grayscale + CLAHE (Preprocessing)
    gray = preprocess_image(img)
    gray_b64 = encode_image_base64(gray)

    # Step 2: Keypoints Detection (Untuk visualisasi web)
    sift = get_sift_detector()
    kp, _ = sift.detectAndCompute(gray, None)
    kp_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_b64 = encode_image_base64(kp_img)
    
    # Step 3 & 4: Jalankan deteksi objek (SIFT -> Matching -> RANSAC -> Cluster)
    results = detect_objects(img)

    out_img = img.copy()
    analysis_details = []

    
    for r in results:
        # Gambar Bounding Box di gambar akhir
        box = np.int32(r["box"])
        color = (0, 255, 0) if r["category"] == 'organik' else (0, 165, 255)
        cv2.polylines(out_img, [box], True, color, 3, cv2.LINE_AA)
        
        # Buat gambar garis-garis pencocokan SIFT (Feature Matching)
        match_img = draw_sift_matches(r, img)
        match_b64 = encode_image_base64(match_img)
        tpl_b64 = encode_image_base64(r['dataset_img'])
        
        analysis_details.append({
            'item': r['item'],
            'category': r['category'],
            'inliers': r['inliers'],
            'confidence': f"{r['confidence']:.1f}%",
            **_hitung_metric_sift(r),
            'kp_template_cnt': len(r['kp_dataset']),
            'template_b64': tpl_b64,
            'match_b64': match_b64
        })
        
    final_b64 = encode_image_base64(out_img)
    
    # Kirim semua gambar Base64 ke frontend HTML
    return render_template('analysis.html', 
                           original_b64=gray_b64,
                           kp_b64=kp_b64,
                           final_b64=final_b64,
                           total_kp=len(kp),
                           details=analysis_details)

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
