"""
check_dataset_quality.py
========================
Script diagnostik untuk mengevaluasi apakah gambar-gambar dalam dataset
cukup baik untuk dikenali oleh sistem SIFT.

Cara pakai:
    python scripts/check_dataset_quality.py

Output:
    - Laporan per objek: jumlah gambar, rata-rata keypoint, skor matching
    - Rekomendasi: BAIK / CUKUP / PERLU PERBAIKAN
    - File laporan: dataset_quality_report.txt
"""

import cv2
import os
import sys
import numpy as np
from itertools import combinations

# Fix encoding Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORT_PATH = os.path.join(BASE_DIR, "dataset_quality_report.txt")
sys.path.insert(0, BASE_DIR)

from src.core.preprocessor import preprocess_image

# ── Konstanta (sama dengan detector.py) ─────────────────────────────────────
RATIO_LOWE       = 0.75
MIN_GOOD_MATCH   = 8
MIN_KP           = 30          # minimal KP agar gambar dianggap "kaya fitur"
MIN_IMG_PER_ITEM = 5           # minimal gambar per objek
SAMPLE_LIMIT     = 15          # batasi gambar yang diproses per objek (hemat waktu)

EKSTENSI_VALID = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ── Warna terminal ───────────────────────────────────────────────────────────
HIJAU  = "\033[92m"
KUNING = "\033[93m"
MERAH  = "\033[91m"
BIRU   = "\033[94m"
RESET  = "\033[0m"
TEBAL  = "\033[1m"


# ============================================================================
# FUNGSI UTILITAS
# ============================================================================

def muat_gambar_dari_folder(folder_path, limit=SAMPLE_LIMIT):
    """Muat gambar valid dari folder, batasi jumlahnya."""
    gambar_list = []
    for nama in sorted(os.listdir(folder_path)):
        if not nama.lower().endswith(EKSTENSI_VALID):
            continue
        path = os.path.join(folder_path, nama)
        img  = cv2.imread(path)
        if img is None:
            continue
        # Resize ke standar agar konsisten
        img = cv2.resize(img, (400, 400))
        gambar_list.append((nama, img))
        if len(gambar_list) >= limit:
            break
    return gambar_list


def ekstrak_fitur(img, sift):
    """Preprocess + ekstrak SIFT. Return (kp, des) atau ([], None)."""
    gray = preprocess_image(img)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) == 0:
        return [], None
    # Ambil top-500 KP terkuat (sesuai build_features.py)
    idx = np.argsort([-k.response for k in kp])[:500]
    kp  = [kp[i] for i in idx]
    des = des[idx]
    return kp, des


def hitung_good_match(des_a, des_b, matcher):
    """Jumlah good match antara dua deskriptor (Lowe Ratio Test)."""
    if des_a is None or des_b is None:
        return 0
    try:
        matches = matcher.knnMatch(des_a, des_b, k=2)
    except Exception:
        return 0
    good = 0
    for pasangan in matches:
        if len(pasangan) < 2:
            continue
        m, n = pasangan
        if m.distance < RATIO_LOWE * n.distance:
            good += 1
    return good


# ============================================================================
# ANALISIS PER OBJEK
# ============================================================================

def analisis_item(item_name, folder_path, sift, matcher):
    """
    Evaluasi kualitas satu folder objek.
    Returns dict dengan semua metrik.
    """
    gambar_list = muat_gambar_dari_folder(folder_path)
    jumlah_img  = len(gambar_list)

    hasil = {
        "item"          : item_name,
        "jumlah_gambar" : jumlah_img,
        "kp_per_gambar" : [],
        "match_scores"  : [],   # intra-class: match antar sesama gambar
        "pasang_diuji"  : 0,
        "gambar_miskin" : [],   # gambar dengan KP < MIN_KP
    }

    if jumlah_img == 0:
        return hasil

    # ── Ekstraksi fitur semua gambar ─────────────────────────────────────
    fitur_list = []
    for nama, img in gambar_list:
        kp, des = ekstrak_fitur(img, sift)
        n_kp    = len(kp)
        hasil["kp_per_gambar"].append(n_kp)
        if n_kp < MIN_KP:
            hasil["gambar_miskin"].append(nama)
        fitur_list.append((nama, kp, des))

    # ── Intra-class matching (uji semua kombinasi pasangan) ──────────────
    pasang = list(combinations(range(len(fitur_list)), 2))
    # Batasi 20 pasang agar tidak terlalu lama
    if len(pasang) > 20:
        idx_acak = np.random.choice(len(pasang), 20, replace=False)
        pasang   = [pasang[i] for i in idx_acak]

    for i, j in pasang:
        _, kp_a, des_a = fitur_list[i]
        _, kp_b, des_b = fitur_list[j]
        good = hitung_good_match(des_a, des_b, matcher)
        hasil["match_scores"].append(good)
        hasil["pasang_diuji"] += 1

    return hasil


# ============================================================================
# PENILAIAN & REKOMENDASI
# ============================================================================

def beri_nilai(hasil):
    """Hitung skor 0-100 dan beri rekomendasi."""
    jml   = hasil["jumlah_gambar"]
    kp    = hasil["kp_per_gambar"]
    ms    = hasil["match_scores"]
    miskin = len(hasil["gambar_miskin"])

    skor = 0

    # 1. Jumlah gambar (bobot 30 poin)
    if jml >= 20:
        skor += 30
    elif jml >= 10:
        skor += 20
    elif jml >= MIN_IMG_PER_ITEM:
        skor += 10
    else:
        skor += 0   # kurang dari MIN_IMG_PER_ITEM

    # 2. Rata-rata keypoint per gambar (bobot 30 poin)
    rata_kp = np.mean(kp) if kp else 0
    if rata_kp >= 300:
        skor += 30
    elif rata_kp >= 150:
        skor += 20
    elif rata_kp >= MIN_KP:
        skor += 10
    # else 0

    # 3. Intra-class match rate (bobot 30 poin)
    if ms:
        rata_match = np.mean(ms)
        lolos_rate = sum(1 for m in ms if m >= MIN_GOOD_MATCH) / len(ms)
        if rata_match >= 20 and lolos_rate >= 0.7:
            skor += 30
        elif rata_match >= 10 and lolos_rate >= 0.5:
            skor += 20
        elif lolos_rate >= 0.3:
            skor += 10

    # 4. Penalti gambar miskin fitur (bobot -10 per 25%)
    if kp:
        pct_miskin = miskin / len(kp)
        skor -= int(pct_miskin * 10)

    skor = max(0, min(100, skor))

    if skor >= 70:
        rekomendasi = f"{HIJAU}[OK] BAIK -- Dataset cocok untuk SIFT{RESET}"
        status = "BAIK"
    elif skor >= 40:
        rekomendasi = f"{KUNING}[!!] CUKUP -- Tambah lebih banyak gambar variasi{RESET}"
        status = "CUKUP"
    else:
        rekomendasi = f"{MERAH}[XX] PERLU PERBAIKAN -- Dataset tidak ideal untuk SIFT{RESET}"
        status = "PERLU PERBAIKAN"

    return skor, rekomendasi, status


# ============================================================================
# MAIN
# ============================================================================

def main():
    sift    = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    print(f"\n{TEBAL}{BIRU}{'='*60}{RESET}")
    print(f"{TEBAL}{BIRU}   DIAGNOSTIK KUALITAS DATASET — NetraDetTrash{RESET}")
    print(f"{TEBAL}{BIRU}{'='*60}{RESET}\n")

    semua_hasil = []

    for kategori in ["organik", "anorganik"]:
        kat_path = os.path.join(DATASET_DIR, kategori)
        if not os.path.isdir(kat_path):
            continue

        print(f"{TEBAL}[KATEGORI] {kategori.upper()}{RESET}")
        print("-" * 60)

        for item_folder in sorted(os.listdir(kat_path)):
            item_path = os.path.join(kat_path, item_folder)
            if not os.path.isdir(item_path):
                continue

            print(f"  >> Menganalisis: {item_folder} ...", end="", flush=True)

            hasil = analisis_item(item_folder, item_path, sift, matcher)
            skor, rekomendasi, status = beri_nilai(hasil)

            hasil["skor"]         = skor
            hasil["rekomendasi"]  = rekomendasi
            hasil["status"]       = status
            hasil["kategori"]     = kategori
            semua_hasil.append(hasil)

            print(f"\r  {'─'*56}")
            print(f"  {TEBAL}Objek   :{RESET} {item_folder}")
            print(f"  Gambar  : {hasil['jumlah_gambar']} file")

            if hasil["kp_per_gambar"]:
                rata_kp    = np.mean(hasil["kp_per_gambar"])
                min_kp     = min(hasil["kp_per_gambar"])
                max_kp     = max(hasil["kp_per_gambar"])
                print(f"  KP/img  : rata={rata_kp:.0f}  min={min_kp}  max={max_kp}")
            else:
                print(f"  KP/img  : -")

            if hasil["match_scores"]:
                rata_m     = np.mean(hasil["match_scores"])
                lolos_rate = sum(1 for m in hasil["match_scores"] if m >= MIN_GOOD_MATCH)
                total_p    = hasil["pasang_diuji"]
                print(f"  Intra-match: rata={rata_m:.1f} good_match  |  "
                      f"lolos={lolos_rate}/{total_p} pasang ({lolos_rate/total_p*100:.0f}%)")
            else:
                print(f"  Intra-match: tidak cukup gambar")

            if hasil["gambar_miskin"]:
                print(f"  {KUNING}[!] {len(hasil['gambar_miskin'])} gambar miskin KP (<{MIN_KP}): "
                      f"{', '.join(hasil['gambar_miskin'][:3])}{'...' if len(hasil['gambar_miskin'])>3 else ''}{RESET}")

            print(f"  Skor    : {TEBAL}{skor}/100{RESET}")
            print(f"  Status  : {rekomendasi}")
            print()

    # ── Ringkasan ─────────────────────────────────────────────────────────
    print(f"\n{TEBAL}{BIRU}{'='*60}{RESET}")
    print(f"{TEBAL}  RINGKASAN{RESET}")
    print(f"{TEBAL}{BIRU}{'='*60}{RESET}")
    print(f"  {'Objek':<28} {'Kat':<12} {'Gambar':>6}  {'Skor':>5}  Status")
    print(f"  {'─'*56}")
    for h in sorted(semua_hasil, key=lambda x: x["skor"]):
        warna = HIJAU if h["status"]=="BAIK" else (KUNING if h["status"]=="CUKUP" else MERAH)
        print(f"  {h['item']:<28} {h['kategori']:<12} {h['jumlah_gambar']:>6}  "
              f"{h['skor']:>5}  {warna}{h['status']}{RESET}")

    # ── Saran khusus organik ─────────────────────────────────────────────
    organik_buruk = [h for h in semua_hasil
                     if h["kategori"] == "organik" and h["status"] != "BAIK"]
    if organik_buruk:
        print(f"\n{TEBAL}{MERAH}SARAN UNTUK OBJEK ORGANIK BERMASALAH:{RESET}")
        print("""
  Objek organik sulit untuk SIFT karena:
  • Bentuk tidak rigid (berubah-ubah)
  • Tekstur homogen (tidak banyak sudut/tepi unik)
  • Variasi antar foto terlalu tinggi

  Solusi yang direkomendasikan:
  ① Tambah gambar dari BANYAK sudut (atas, samping, diagonal) — minimal 20 foto
  ② Pastikan latar belakang KONTRAS dengan objek (pakai alas putih/hitam polos)
  ③ Foto harus tajam & terang — hindari blur dan bayangan keras
  ④ Untuk objek organik: fokus pada SATU area fitur unik
     - Cangkang telur → foto detail area RETAKAN
     - Kulit pisang   → foto detail area BINTIK COKELAT
  ⑤ Jika setelah semua itu skor masih rendah, pertimbangkan:
     - Mengganti dengan klasifikasi CNN ringan (MobileNetV2)
     - Atau menggunakan ResNet50 pretrained + fine-tune dengan dataset ini
        """)

    # ── Simpan laporan ke file ────────────────────────────────────────────
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("LAPORAN KUALITAS DATASET — NetraDetTrash\n")
        f.write("=" * 60 + "\n\n")
        for h in semua_hasil:
            f.write(f"Objek    : {h['item']} ({h['kategori']})\n")
            f.write(f"Gambar   : {h['jumlah_gambar']}\n")
            if h["kp_per_gambar"]:
                f.write(f"KP rata  : {np.mean(h['kp_per_gambar']):.0f}\n")
            if h["match_scores"]:
                f.write(f"Match rata: {np.mean(h['match_scores']):.1f}\n")
            f.write(f"Skor     : {h['skor']}/100\n")
            f.write(f"Status   : {h['status']}\n")
            f.write("-" * 40 + "\n")

    print(f"\n[LAPORAN] Disimpan ke: {REPORT_PATH}\n")


if __name__ == "__main__":
    main()
