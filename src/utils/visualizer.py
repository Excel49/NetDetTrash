import cv2
import numpy as np
import base64
from src.core.detector import detect_objects

def encode_image_base64(img):
    """Mengonversi gambar array OpenCV menjadi string Base64 untuk ditampilkan di HTML."""
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buffer).decode('utf-8')

def draw_roi_visualization(img, rois):
    """Menggambar kotak ROI ke atas gambar untuk visualisasi preprocessing."""
    vis_img = img.copy()
    for (x, y, w, h) in rois:
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 255), 3) # Kotak Kuning
    return vis_img

def draw_bounding_boxes(img, results):
    out_img = img.copy()
    overlay = out_img.copy()
    
    for r in results:
        box = np.int32(r["box"])
        # Hijau untuk organik, Oranye untuk anorganik
        color = (0, 255, 0) if r["category"] == 'organik' else (0, 165, 255)
        
        cv2.polylines(out_img, [box], True, color, 3, cv2.LINE_AA)
        
        x_min = int(np.min(box[:, 0, 0]))
        y_min = int(np.min(box[:, 0, 1]))
        
        label_category = r['category'].capitalize()
        conf_str = f"{r['confidence']:.1f}%"
        label_item = f"{r['item']}  [{r['inliers']} inliers | {conf_str}]"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        
        (tw1, th1), _ = cv2.getTextSize(label_category, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(label_item, font, font_scale - 0.1, thickness - 1)
        
        pad = 6
        box_w = max(tw1, tw2) + pad * 2
        box_h = th1 + th2 + pad * 3
        label_y_top = max(0, y_min - box_h - 4)
        
        cv2.rectangle(overlay, (x_min, label_y_top), (x_min + box_w, label_y_top + box_h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, out_img, 0.3, 0, out_img)
        overlay = out_img.copy()
        
        cv2.rectangle(out_img, (x_min, label_y_top), (x_min + box_w, label_y_top + box_h), color, 1)
        cv2.putText(out_img, label_category, (x_min + pad, label_y_top + pad + th1), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(out_img, label_item, (x_min + pad, label_y_top + pad * 2 + th1 + th2), font, font_scale - 0.1, (220, 220, 220), thickness - 1, cv2.LINE_AA)
        
    return out_img

def draw_sift_matches(result_item, input_img):
    match_img = cv2.drawMatches(
        result_item['dataset_img'], result_item['kp_dataset'],
        input_img, result_item['kp_input'],
        result_item['matches'], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return match_img

def draw_ransac_inliers_outliers(result_item, input_img):
    """Menggambar inliers (garis hijau) dan outliers (garis merah) yang dibuang RANSAC."""
    match_img = cv2.drawMatches(
        result_item['dataset_img'], result_item['kp_dataset'],
        input_img, result_item['kp_input'],
        result_item['matches'], None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    if 'outliers' in result_item and result_item['outliers']:
        match_img = cv2.drawMatches(
            result_item['dataset_img'], result_item['kp_dataset'],
            input_img, result_item['kp_input'],
            result_item['outliers'], match_img,
            matchColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
        )
    return match_img

def process_classification_result(img, results):
    if not results:
        return {
            'category': 'Tidak Dikenali',
            'item': '-',
            'inliers': 0,
            'image_base64': encode_image_base64(img),
            'matches': []
        }
        
    matches_detail = []
    for r in results:
        match_img = draw_sift_matches(r, img)
        matches_detail.append({
            'item': r['item'],
            'category': r['category'],
            'inliers': r['inliers'],
            'confidence': f"{r['confidence']:.1f}%",
            'kp_dataset': len(r['kp_dataset']),
            'template_base64': encode_image_base64(r['dataset_img']),
            'match_image_base64': encode_image_base64(match_img)
        })
        
    categories = list(set([r["category"].capitalize() for r in results]))
    items = [f"{r['item']} ({r['confidence']:.1f}%)" for r in results]
    
    out_img = draw_bounding_boxes(img, results)
    
    return {
        'category': ", ".join(categories),
        'item': " | ".join(items),
        'inliers': sum([r["inliers"] for r in results]),
        'image_base64': encode_image_base64(out_img),
        'matches': matches_detail
    }

def generate_frames():
    """Generator untuk stream video realtime (M-JPEG)"""
    camera = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Deteksi objek pada setiap frame
            results = detect_objects(frame)
            
            # Gambar bounding box jika ada hasil deteksi
            if results:
                frame = draw_bounding_boxes(frame, results)
                
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()
