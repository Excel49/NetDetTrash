import cv2
import matplotlib.pyplot as plt

# =========================
# BACA GAMBAR
# =========================
image_path = "telur.png"   # ganti dengan nama file kamu

img = cv2.imread(image_path)

# cek gambar berhasil dibaca
if img is None:
    print("Gambar tidak ditemukan!")
    exit()

# =========================
# RESIZE OPTIONAL
# =========================
img = cv2.resize(img, (500, 500))

# =========================
# PREPROCESSING
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

gray = clahe.apply(gray)

# =========================
# SIFT
# =========================
sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Jumlah Keypoints :", len(keypoints))

# =========================
# GAMBAR KEYPOINT
# =========================
result = cv2.drawKeypoints(
    img,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# =========================
# TAMPILKAN HASIL
# =========================
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints")
plt.axis("off")
plt.show()