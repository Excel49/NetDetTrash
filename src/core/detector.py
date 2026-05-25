import cv2
import numpy as np

from src.core.dataset import get_dataset_cache, get_sift_detector
from src.core.preprocessor import preprocess_image

"""
detector.py
Ringkasan alur (singkat):
1. Ambil cache dataset + SIFT detector
2. Preprocess input (CLAHE + blur)
3. Ekstrak keypoints/deskriptor dengan SIFT
4. Untuk tiap template: lakukan BFMatcher + Lowe ratio
5. Untuk matches yang cukup: jalankan RANSAC/homography untuk dapat inlier
6. Validasi geometri, buat bounding box, dan gabung deteksi serupa
7. Terapkan NMS + filter kualitas, kembalikan list hasil

Perubahan yang dibuat: menambahkan docstring alur dan menghapus print debug
tanpa mengubah perilaku fungsi asli.
"""

matcher_bf = cv2.BFMatcher()
MIN_GOOD_MATCH = 8
MIN_INLIER = 10
RATIO_LOWE = 0.75
MAKS_OBJECT_PER_TEMPLATE = 5
MIN_TEMPLATE_COVERAGE = 0.05
MIN_INPUT_COVERAGE = 0.005
MIN_INLIER_RATIO = 0.15

def buat_box_dari_keypoint(titik_inlier, lebar_gambar, tinggi_gambar, padding=20):
    """Bounding box dari cluster keypoint."""
    titik = titik_inlier.reshape(-1, 2)
    x_min, y_min = np.min(titik, axis=0)
    x_max, y_max = np.max(titik, axis=0)

    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(lebar_gambar - 1, int(x_max) + padding)
    y_max = min(tinggi_gambar - 1, int(y_max) + padding)

    return np.float32([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_max],
        [x_max, y_min],
    ]).reshape(-1, 1, 2)

def hitung_iou(box_a, box_b):
    """Cek overlap antar box."""
    ax1, ay1 = box_a[:, 0, 0].min(), box_a[:, 0, 1].min()
    ax2, ay2 = box_a[:, 0, 0].max(), box_a[:, 0, 1].max()
    bx1, by1 = box_b[:, 0, 0].min(), box_b[:, 0, 1].min()
    bx2, by2 = box_b[:, 0, 0].max(), box_b[:, 0, 1].max()

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    luas_inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    luas_a = (ax2 - ax1) * (ay2 - ay1)
    luas_b = (bx2 - bx1) * (by2 - by1)
    luas_union = luas_a + luas_b - luas_inter
    return luas_inter / luas_union if luas_union > 0 else 0

def nms_sederhana(hasil_deteksi, batas_iou=0.3):
    """Buang box yang dobel."""
    hasil_bersih = []
    for kandidat in sorted(hasil_deteksi, key=lambda x: x["inliers"], reverse=True):
        dobel = False
        for diterima in hasil_bersih:
            if hitung_iou(kandidat["box"], diterima["box"]) > batas_iou:
                dobel = True
                break
        if not dobel:
            hasil_bersih.append(kandidat)
    return hasil_bersih

def info_box(box):
    """Ambil posisi sederhana dari box."""
    x1, y1 = box[:, 0, 0].min(), box[:, 0, 1].min()
    x2, y2 = box[:, 0, 0].max(), box[:, 0, 1].max()
    tengah_x = (x1 + x2) / 2
    tengah_y = (y1 + y2) / 2
    lebar = x2 - x1
    tinggi = y2 - y1
    return tengah_x, tengah_y, lebar, tinggi

def hitung_coverage(points_xy, lebar, tinggi):
    """Luas sebaran titik dibanding luas gambar."""
    if points_xy.size == 0:
        return 0.0
    x_min, y_min = np.min(points_xy, axis=0)
    x_max, y_max = np.max(points_xy, axis=0)
    # pastikan tidak nol untuk menghindari pembagian 0
    area_cluster = max(1.0, float((x_max - x_min) * (y_max - y_min)))
    area_total = max(1.0, float(lebar * tinggi))
    return area_cluster / area_total

def lolos_validasi_geometri(data_template, titik_template_inlier, titik_input_inlier, frame_w, frame_h, jumlah_inlier, jumlah_good_match):
    """Filter geometri agar false match berkurang."""
    if jumlah_good_match <= 0:
        return False

    rasio_inlier = jumlah_inlier / jumlah_good_match
    if rasio_inlier < MIN_INLIER_RATIO:
        return False

    tpl_h = max(1, data_template["height"])
    tpl_w = max(1, data_template["width"])
    coverage_template = hitung_coverage(titik_template_inlier.reshape(-1, 2), tpl_w, tpl_h)
    if coverage_template < MIN_TEMPLATE_COVERAGE:
        return False

    coverage_input = hitung_coverage(titik_input_inlier.reshape(-1, 2), frame_w, frame_h)
    if coverage_input < MIN_INPUT_COVERAGE:
        return False

    return True

def gabung_deteksi_serupa(hasil_deteksi):
    """Gabung deteksi item yang sama dan posisinya dekat."""
    hasil_bersih = []

    for kandidat in sorted(hasil_deteksi, key=lambda x: x["inliers"], reverse=True):
        kx, ky, kw, kh = info_box(kandidat["box"])
        dobel = False

        for diterima in hasil_bersih:
            if kandidat["item"] != diterima["item"]:
                continue

            dx, dy, dw, dh = info_box(diterima["box"])
            jarak_tengah = np.hypot(kx - dx, ky - dy)
            ukuran_acuan = max(min(kw, dw), min(kh, dh), 1)
            overlap = hitung_iou(kandidat["box"], diterima["box"])

            #overlap
            if overlap > 0.3 or jarak_tengah < ukuran_acuan * 0.3:
                dobel = True
                break

        if not dobel:
            hasil_bersih.append(kandidat)

    return hasil_bersih

def cari_object_berulang(data_template, good_match, kp_input, lebar_gambar, tinggi_gambar):
    """Iterative RANSAC untuk banyak objek."""
    hasil_template = []
    sisa_match = good_match[:]

    for _ in range(MAKS_OBJECT_PER_TEMPLATE):
        if len(sisa_match) < MIN_GOOD_MATCH:
            break

        titik_template = np.float32([data_template["kp"][m.queryIdx].pt for m in sisa_match]).reshape(-1, 1, 2)
        titik_input = np.float32([kp_input[m.trainIdx].pt for m in sisa_match]).reshape(-1, 1, 2)

        # RANSAC / Homography
        homography, mask = cv2.findHomography(titik_template, titik_input, cv2.RANSAC, 5.0)
        if homography is None or mask is None:
            break

        mask_inlier = mask.ravel() == 1
        jumlah_inlier = int(np.sum(mask_inlier))
        if jumlah_inlier < MIN_INLIER:
            break

        # Cluster keypoint dari inlier
        titik_template_inlier = titik_template[mask_inlier]
        titik_cluster = titik_input[mask_inlier]
        if not lolos_validasi_geometri(
            data_template,
            titik_template_inlier,
            titik_cluster,
            lebar_gambar,
            tinggi_gambar,
            jumlah_inlier,
            len(sisa_match)
        ):
            # Hasil ini tidak valid, buang inliernya agar iterasi lanjut cari cluster lain
            sisa_match = [m for i, m in enumerate(sisa_match) if not mask_inlier[i]]
            continue

        box_object = buat_box_dari_keypoint(titik_cluster, lebar_gambar, tinggi_gambar)
        match_inlier = [sisa_match[i] for i, ok in enumerate(mask_inlier) if ok]
        confidence = min(99.9, 45.0 + jumlah_inlier * 1.8)

        hasil_template.append({
            "category": data_template["category"],
            "item": data_template["item"],
            "inliers": jumlah_inlier,
            "confidence": confidence,
            "box": box_object,
            "dataset_img": data_template["image"],
            "kp_dataset": data_template["kp"],
            "kp_input": kp_input,
            "matches": match_inlier
        })

        # Hapus inlier, cari objek berikutnya dari sisa match
        sisa_match = [m for i, m in enumerate(sisa_match) if not mask_inlier[i]]

    return hasil_template

def detect_objects(frame):
    dataset_cache = get_dataset_cache()
    sift = get_sift_detector()
    tinggi_gambar, lebar_gambar = frame.shape[:2]

    # Input Image -> CLAHE -> Gaussian blur ringan
    gambar_proses = preprocess_image(frame)

    # SIFT detectAndCompute
    kp_input, des_input = sift.detectAndCompute(gambar_proses, None)
    if des_input is None or len(kp_input) < MIN_GOOD_MATCH:
        return []

    hasil_deteksi = []

    for data_template in dataset_cache:
        try:
            semua_match = matcher_bf.knnMatch(data_template["des"], des_input, k=2)
        except Exception:
            continue

        # Lowe Ratio Test
        good_match = []
        for pasangan in semua_match:
            if len(pasangan) < 2:
                continue
            match_terbaik = pasangan[0]
            match_kedua = pasangan[1]
            if match_terbaik.distance < RATIO_LOWE * match_kedua.distance:
                good_match.append(match_terbaik)

        if len(good_match) < MIN_GOOD_MATCH:
            continue

        hasil_deteksi.extend(
            cari_object_berulang(data_template, good_match, kp_input, lebar_gambar, tinggi_gambar)
        )

    if not hasil_deteksi:
        return []

    # Buang deteksi yang terlalu lemah dari hasil terbaik
    inlier_terbaik = max(hasil["inliers"] for hasil in hasil_deteksi)
    batas_inlier = max(MIN_INLIER, int(inlier_terbaik * 0.10))
    hasil_deteksi = [hasil for hasil in hasil_deteksi if hasil["inliers"] >= batas_inlier]

    hasil_akhir = nms_sederhana(hasil_deteksi)
    hasil_akhir = gabung_deteksi_serupa(hasil_akhir)

    # Filter: jika inliers < 20, anggap tidak terdeteksi
    hasil_akhir = [hasil for hasil in hasil_akhir if hasil["inliers"] >= 10]

    # debug prints dihilangkan agar output lebih ringkas; perilaku deteksi tetap sama

    return hasil_akhir
