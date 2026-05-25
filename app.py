from flask import Flask
from src.api.routes import main_bp
from src.core.dataset import load_dataset_to_memory

app = Flask(__name__)

# Daftarkan rute-rute API yang sudah dibuat
app.register_blueprint(main_bp)

if __name__ == '__main__':
    # Eksekusi pemuatan dataset/model SIFT ke memori SEBELUM server jalan.
    # (Pastikan Anda sudah menjalankan scripts/build_features.py sebelumnya)
    try:
        load_dataset_to_memory()
    except Exception as e:
        print(f"Peringatan: {e}")
        print("Sistem akan mencoba memuat model saat request pertama masuk.")
        
    app.run(debug=True, host='0.0.0.0')