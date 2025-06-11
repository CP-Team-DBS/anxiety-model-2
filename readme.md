API Analisis Journaling Emosional
API ini adalah sebuah sistem lengkap yang dirancang untuk menganalisis teks dari sebuah jurnal, mendeteksi emosi yang terkandung di dalamnya, dan memberikan umpan balik berupa insight, validasi, dan saran yang relevan.

Sistem ini menggunakan dua model machine learning yang bekerja secara berurutan:

Model 1 (Deteksi Emosi): Sebuah model TensorFlow Lite yang menerima input teks, menerjemahkannya, dan memprediksi skor untuk enam emosi: anxiety, fear, nervousness, sadness, suffering, dan shame. Hasilnya kemudian dikonversi menjadi 3 emosi paling dominan dengan level 0-3.

Model 2 (Retrieval): Sebuah model berbasis TF-IDF dan Cosine Similarity yang menerima hasil dari Model 1 untuk mencari dan mengambil set respons (insight, validation, saran) yang paling cocok dari database.

1. Struktur Folder Proyek
Untuk menjalankan API ini, pastikan Anda memiliki struktur folder dan file sebagai berikut. Semua perintah di terminal dijalankan dari folder root (capstone_project_ml).

capstone_project_ml/
│
├── api/
│   ├── main_api.py                      # Skrip utama FastAPI
│   ├── emotion_regression_model_v2.tflite  # Aset Model 1
│   ├── tokenizer_config.json          # Aset Model 1 (dibuat)
│   ├── retrieval_database.csv         # Aset Model 2
│   ├── vectorizer_config.json         # Aset Model 2 (dibuat)
│   └── tfidf_matrix.npz               # Aset Model 2 (dibuat)
│
├── buat_tokenizer.py                    # Skrip untuk membuat tokenizer_config.json
├── buat_model_retrieval.py            # Skrip untuk membuat aset Model 2
├── dataset.csv                          # Dataset training asli untuk Model 1
└── requirements.txt                     # File dependensi Python

2. Persiapan Aset Model
Sebelum menjalankan API, Anda harus membuat file-file aset yang dibutuhkan oleh model.

2.1. Membuat Aset Tokenizer untuk Model 1
Jalankan skrip ini untuk membuat tokenizer_config.json dari dataset training Anda.

# Jalankan dari folder root (capstone_project_ml)
python buat_tokenizer.py

Setelah selesai, pindahkan file tokenizer_config.json yang baru dibuat ke dalam folder api.

2.2. Membuat Aset untuk Model Retrieval (Model 2)
Jalankan skrip ini untuk membuat vectorizer_config.json dan tfidf_matrix.npz.

# Jalankan dari folder root (capstone_project_ml)
python buat_model_retrieval.py

Setelah selesai, pindahkan kedua file yang baru dibuat (vectorizer_config.json dan tfidf_matrix.npz) ke dalam folder api.

3. Instalasi dan Menjalankan API
3.1. Buat Virtual Environment
Ini adalah praktik terbaik untuk menjaga dependensi proyek tetap terisolasi.

# Buat environment baru bernama 'venv'
python -m venv venv

# Aktifkan environment
# Untuk Windows:
.\venv\Scripts\activate
# Untuk macOS/Linux:
# source venv/bin/activate

3.2. Instal Dependensi
Pastikan venv Anda sudah aktif, lalu instal semua library yang dibutuhkan dari file requirements.txt.

# Jalankan dari folder root (capstone_project_ml)
pip install -r requirements.txt

3.3. Jalankan Server API
Setelah semua aset siap dan dependensi terinstal, jalankan server API menggunakan uvicorn.

# Jalankan dari folder root (capstone_project_ml)
uvicorn api.main_api:app --reload

Server akan berjalan di http://127.0.0.1:8000.

4. Cara Menggunakan API
4.1. Akses Dokumentasi Interaktif
Buka browser web Anda dan kunjungi alamat berikut untuk melihat dokumentasi API yang dibuat secara otomatis oleh FastAPI.
http://127.0.0.1:8000/docs

4.2. Mengirim Permintaan
Di halaman dokumentasi, buka endpoint POST /process-journal/.

Klik tombol "Try it out".

Masukkan teks jurnal Anda di dalam Request body.

Contoh Request Body:

{
  "journal_text": "Besok aku akan melakukan sempro, aku merasa sangat gugup. Ini membuatku kepikiran terus."
}

Klik tombol "Execute".

4.3. Contoh Respons
Jika berhasil, Anda akan menerima respons JSON seperti berikut:

{
  "predicted_emotions": {
    "anxiety": 2,
    "fear": 0,
    "nervousness": 3,
    "sadness": 0,
    "suffering": 0,
    "shame": 0
  },
  "insight": "cemas lumayan, gelisah tinggi",
  "validation": "Duh, kamu lagi merasa cemas dan gelisah banget ya. Wajar kok merasa begitu, apalagi akan menghadapi momen penting seperti sempro.",
  "saran": "1. Coba tarik napas dalam-dalam selama beberapa menit untuk menenangkan diri.\r\n2. Fokus pada persiapan yang sudah kamu lakukan, percayalah pada dirimu sendiri.\r\n3. Ingat, perasaan ini akan berlalu. Kamu pasti bisa melaluinya!"
}
