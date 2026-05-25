# Penjelasan Detail Flowchart NetraDetTrash

> Setiap tahap dijelaskan menggunakan **matriks 5×5 piksel** dengan angka konkret
> dan **perhitungan matematis** yang sesuai dengan implementasi kode.

---

## TAHAP 1 — Input Pengguna

Sistem menerima gambar berwarna (BGR). Anggap gambar kita adalah **patch 5×5 piksel**
dari sebuah botol Aqua, di mana setiap sel = nilai piksel [B, G, R]:

```
Gambar Input BGR (5×5):
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ [180,210,230]│ [175,205,225]│ [160,195,220]│ [170,200,215]│ [165,190,210]│
│ [200,220,240]│ [185,215,235]│ [155,190,215]│ [180,210,230]│ [170,200,220]│
│ [150,180,200]│ [195,225,245]│ [210,240,255]│ [160,190,210]│ [175,205,225]│
│ [165,195,215]│ [170,200,220]│ [185,215,235]│ [200,230,250]│ [155,185,205]│
│ [180,210,230]│ [160,190,210]│ [175,205,225]│ [165,195,215]│ [190,220,240]│
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## TAHAP 2 — Preprocessing

### 2a. Konversi Grayscale

**Rumus:**
```
Y = 0.299·R + 0.587·G + 0.114·B
```

**Contoh perhitungan sel (baris 0, kolom 0) → [B=180, G=210, R=230]:**
```
Y = 0.299 × 230 + 0.587 × 210 + 0.114 × 180
  = 68.77 + 123.27 + 20.52
  = 212.56  →  dibulatkan  213
```

**Hasil matriks Grayscale (5×5):**
```
┌─────┬─────┬─────┬─────┬─────┐
│ 213 │ 208 │ 203 │ 205 │ 200 │
│ 224 │ 219 │ 198 │ 213 │ 208 │
│ 188 │ 230 │ 248 │ 198 │ 213 │
│ 203 │ 208 │ 219 │ 235 │ 193 │
│ 213 │ 198 │ 208 │ 203 │ 224 │
└─────┴─────┴─────┴─────┴─────┘
Min = 188,  Max = 248,  Range = 60
```

---

### 2b. CLAHE — Contrast Limited Adaptive Histogram Equalization

CLAHE membagi gambar ke tile 8×8 dan melakukan *histogram equalization* lokal
dengan batas kontras (`clipLimit = 2.0`) agar tidak over-amplify noise.

**Formula histogram equalization:**
```
I_out = round( (I_in - Min) / (Max - Min) × 255 )
```

**Contoh sel (baris 2, kolom 2) → nilai = 248:**
```
I_out = round( (248 - 188) / (248 - 188) × 255 )
      = round( 60 / 60 × 255 )
      = 255
```

**Contoh sel (baris 0, kolom 4) → nilai = 200:**
```
I_out = round( (200 - 188) / 60 × 255 )
      = round( 12 / 60 × 255 )
      = round( 51.0 )
      = 51
```

**Hasil matriks setelah CLAHE (5×5):**
```
┌─────┬─────┬─────┬─────┬─────┐
│ 212 │ 170 │ 128 │ 145 │  51 │
│ 255 │ 212 │  42 │ 212 │ 170 │
│   0 │ 255 │ 255 │  42 │ 212 │
│ 128 │ 170 │ 212 │ 255 │  21 │
│ 212 │  42 │ 170 │ 128 │ 255 │
└─────┴─────┴─────┴─────┴─────┘
```

---

### 2c. Gaussian Blur (kernel 3×3, σ ≈ 0)

**Kernel Gaussian 3×3:**
```
      1   2   1
K = ─── × 2   4   2
     16   1   2   1
```

**Contoh: hitung nilai piksel tengah (baris 2, kolom 2) pada matriks CLAHE:**

Tetangga 3×3 di sekitar (2,2):
```
42  212  42
170 255 212
212 255 255
```

Hasil konvolusi:
```
= (1×42  + 2×212 + 1×42  +
   2×170 + 4×255 + 2×212 +
   1×212 + 2×255 + 1×255) / 16

= (42 + 424 + 42 + 340 + 1020 + 424 + 212 + 510 + 255) / 16
= 3269 / 16
= 204.3  →  204
```

**Hasil akhir preprocessing (5×5) — siap untuk SIFT:**
```
┌─────┬─────┬─────┬─────┬─────┐
│ 185 │ 162 │ 135 │ 120 │  80 │
│ 210 │ 195 │ 160 │ 175 │ 145 │
│ 180 │ 210 │ 204 │ 180 │ 165 │
│ 155 │ 175 │ 195 │ 210 │ 140 │
│ 160 │ 130 │ 160 │ 155 │ 185 │
└─────┴─────┴─────┴─────┴─────┘
```

---

## TAHAP 3 — Ekstraksi Fitur SIFT

SIFT mencari lokasi dengan **variasi gradien tinggi** (sudut, tepi, tekstur).
Tiap keypoint menghasilkan **deskriptor 128-dimensi**.

Misalkan dari gambar input ditemukan **3 keypoint:**

```
KP Input:
  kp[0] → koordinat (x=1, y=1),  size=3.2,  angle=45°
  kp[1] → koordinat (x=3, y=2),  size=2.8,  angle=120°
  kp[2] → koordinat (x=2, y=4),  size=3.5,  angle=270°
```

Deskriptor setiap keypoint = vektor 128-dimensi (disederhanakan menjadi 4-dim):
```
des_input[0] = [0.12, 0.45, 0.31, 0.08, ...]   ← 128 nilai
des_input[1] = [0.55, 0.22, 0.10, 0.63, ...]
des_input[2] = [0.08, 0.71, 0.44, 0.19, ...]
```

---

## TAHAP 4 — Feature Matching & Lowe's Ratio Test

### 4a. BFMatcher kNN (k=2)

Untuk setiap deskriptor **template**, cari 2 deskriptor input terdekat
menggunakan **jarak Euclidean:**

```
d(A, B) = √Σ (A_i - B_i)²
```

**Contoh: des_template[0] vs des_input:**
```
des_template[0] = [0.10, 0.50, 0.30, 0.10]
des_input[0]    = [0.12, 0.45, 0.31, 0.08]
des_input[1]    = [0.55, 0.22, 0.10, 0.63]

d(template[0], input[0]) = √[(0.10-0.12)² + (0.50-0.45)² + (0.30-0.31)² + (0.10-0.08)²]
                         = √[0.0004 + 0.0025 + 0.0001 + 0.0004]
                         = √0.0034
                         = 0.058   ← match terbaik (m)

d(template[0], input[1]) = √[(0.10-0.55)² + (0.50-0.22)² + (0.30-0.10)² + (0.10-0.63)²]
                         = √[0.2025 + 0.0784 + 0.0400 + 0.2809]
                         = √0.6018
                         = 0.776   ← match kedua (n)
```

### 4b. Lowe's Ratio Test

**Syarat:** `m.distance < 0.75 × n.distance`

```
0.058 < 0.75 × 0.776
0.058 < 0.582   ✅  → match DITERIMA (good_match)
```

**Contoh match yang DITOLAK:**
```
m.distance = 0.520
n.distance = 0.610

0.520 < 0.75 × 0.610
0.520 < 0.458   ❌  → match DIBUANG (ambigu)
```

**Misalkan hasil akhir:** `good_match = [m0, m1, m2, m3, m4, m5, m6, m7, m8]`
→ 9 match lolos, ≥ batas MIN_GOOD_MATCH (8) → **lanjut ke RANSAC**

---

## TAHAP 5 — Iterative RANSAC

### Iterasi ke-1: Deteksi Objek Pertama

Koordinat titik pasangan dari `good_match` (9 pasang):

```
 #   Titik Template (x,y)    Titik Input (x,y)
─────────────────────────────────────────────────
m0   (10, 15)                (52, 48)
m1   (25, 10)                (67, 43)
m2   (30, 20)                (72, 53)
m3   (12, 30)                (54, 63)    ← inlier
m4   (20, 25)                (62, 58)    ← inlier
m5   (18, 18)                (60, 51)    ← inlier
m6   (28, 12)                (70, 45)    ← inlier
m7   (90, 80)                (10, 190)   ← OUTLIER (tidak konsisten)
m8   (5,  40)                (130, 5)    ← OUTLIER (tidak konsisten)
```

RANSAC menghitung **Homography Matrix H** yang paling banyak menjelaskan pasangan titik.
Titik yang konsisten dengan H disebut **inlier**, sisanya **outlier**.

**Proyeksi titik template ke input via H (disederhanakan):**
```
[x_input]   [h11 h12 h13]   [x_template]
[y_input] = [h21 h22 h23] × [y_template]
[  1    ]   [h31 h32 h33]   [     1    ]
```

Setelah RANSAC menemukan H terbaik:
- **Inlier:** m0, m1, m2, m3, m4, m5, m6  → **7 inlier**
- **Outlier:** m7, m8

**Validasi Geometri:**
```
inlier_ratio = 7 / 9 = 0.778  ≥ MIN_INLIER_RATIO (0.15)  ✅
coverage_template = area_inlier / area_template = 0.12     ≥ 0.05  ✅
coverage_input    = area_inlier / area_input    = 0.018    ≥ 0.005 ✅
```
→ **Objek ke-1 VALID**

**Hitung Confidence:**
```
confidence = min(99.9, 45.0 + inlier × 1.8)
           = min(99.9, 45.0 + 7 × 1.8)
           = min(99.9, 45.0 + 12.6)
           = min(99.9, 57.6)
           = 57.6%
```

### Setelah Iterasi ke-1: Hapus Inlier

```
sisa_match = [m7, m8]   ← hanya 2 match tersisa
len(sisa_match) = 2  <  MIN_GOOD_MATCH (8)  → STOP
```

→ Hanya **1 objek** terdeteksi dari template ini.

*Jika ada 2 botol, iterasi ke-2 akan menemukan 8+ match lagi dari sisa_match.*

---

## TAHAP 6 — Bounding Box AABB

Ambil koordinat input dari **7 inlier**: (52,48), (67,43), (72,53), (54,63), (62,58), (60,51), (70,45)

**Cari koordinat ekstrem:**
```
x_min = min(52, 67, 72, 54, 62, 60, 70) = 52
x_max = max(52, 67, 72, 54, 62, 60, 70) = 72
y_min = min(48, 43, 53, 63, 58, 51, 45) = 43
y_max = max(48, 43, 53, 63, 58, 51, 45) = 63
```

**Tambahkan padding 20 piksel:**
```
x_min = max(0, 52 - 20) = 32
x_max = min(W-1, 72 + 20) = 92
y_min = max(0, 43 - 20) = 23
y_max = min(H-1, 63 + 20) = 83
```

**Hasil bounding box:**
```
Pojok kiri-atas  : (32, 23)
Pojok kiri-bawah : (32, 83)
Pojok kanan-bawah: (92, 83)
Pojok kanan-atas : (92, 23)

Lebar box  = 92 - 32 = 60 piksel
Tinggi box = 83 - 23 = 60 piksel
```

---

## TAHAP 7 — Post-Processing: NMS & Gabung Deteksi Serupa

### Intersection over Union (IoU)

Misalkan dari template lain ada kandidat Box B yang overlap dengan Box A:

```
Box A (objek terdeteksi):  x1=32, y1=23, x2=92, y2=83  (inlier=7)
Box B (kandidat lain):     x1=50, y1=40, x2=110, y2=100 (inlier=4)
```

**Hitung intersection:**
```
ix1 = max(32, 50)  = 50
iy1 = max(23, 40)  = 40
ix2 = min(92, 110) = 92
iy2 = min(83, 100) = 83

luas_intersection = (92-50) × (83-40) = 42 × 43 = 1806
```

**Hitung union:**
```
luas_A = (92-32) × (83-23) = 60 × 60 = 3600
luas_B = (110-50) × (100-40) = 60 × 60 = 3600

luas_union = 3600 + 3600 - 1806 = 5394
```

**IoU:**
```
IoU = luas_intersection / luas_union
    = 1806 / 5394
    = 0.335
```

**Keputusan NMS:**
```
IoU = 0.335  >  batas_iou (0.3)  → Box B DIBUANG (duplikat)
```
Box A dipertahankan karena memiliki inlier lebih banyak (7 > 4).

---

## TAHAP 8 — Klasifikasi & Output

Setiap deteksi sudah memiliki label `category` dari dataset:

```
Hasil Deteksi Akhir:
┌────────────┬────────────┬─────────┬─────────────┬────────────────┐
│ Item       │ Category   │ Inliers │ Confidence  │ Bounding Box   │
├────────────┼────────────┼─────────┼─────────────┼────────────────┤
│ BotolAqua  │ anorganik  │    7    │   57.6%     │ (32,23)→(92,83)│
└────────────┴────────────┴─────────┴─────────────┴────────────────┘
```

**Estimasi Metrik:**
```
precision = confidence / 100 = 57.6 / 100 = 0.576

kp_dataset = jumlah KP template = 150 (contoh)
recall     = inlier / (inlier + (kp_dataset - inlier) × 0.08)
           = 7 / (7 + (150 - 7) × 0.08)
           = 7 / (7 + 11.44)
           = 7 / 18.44
           = 0.380

F1-Score = 2 × (precision × recall) / (precision + recall)
         = 2 × (0.576 × 0.380) / (0.576 + 0.380)
         = 2 × 0.219 / 0.956
         = 0.458 / 0.956
         = 0.479  →  47.9%
```

**Output akhir dikirim ke browser:**
```json
{
  "category":  "Anorganik",
  "item":      "BotolAqua (57.6%)",
  "inliers":   7,
  "image_base64": "<gambar dengan bounding box oranye>",
  "summary_metrics": {
    "precision": "57.6%",
    "recall":    "38.0%",
    "f1_score":  "47.9%"
  }
}
```

---

## Ringkasan Alur Lengkap

```
Input Gambar
    │
    ▼
[Grayscale]  Y = 0.299R + 0.587G + 0.114B
    │
    ▼
[CLAHE]      I_out = (I_in - Min)/(Max - Min) × 255
    │
    ▼
[Gauss Blur] Konvolusi dengan kernel 3×3 (σ ringan)
    │
    ▼
[SIFT]       detectAndCompute → keypoints + des[128-dim]
    │
    ▼
[kNN Match]  d(A,B) = √Σ(Aᵢ - Bᵢ)²   untuk tiap pasang des
    │
    ▼
[Lowe Test]  m.dist < 0.75 × n.dist  → good_match
    │
    ▼  (loop tiap template)
[Iterative RANSAC]
    ├─ findHomography → inlier/outlier
    ├─ jumlah_inlier ≥ 10?  → valid
    ├─ Hapus inlier, ulangi dari sisa_match
    └─ Stop jika sisa < 8 atau sudah 5 objek
    │
    ▼
[AABB]       x_min/max, y_min/max dari inlier + padding 20px
    │
    ▼
[Confidence] 45.0 + inlier × 1.8  (max 99.9)
    │
    ▼
[NMS]        IoU = intersection/union  >  0.3 → buang duplikat
    │
    ▼
[Output]     Gambar + label + precision/recall/F1
```
