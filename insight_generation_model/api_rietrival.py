# main.py
# Impor library yang dibutuhkan
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# --- LANGKAH 1: KELAS MODEL RETRIEVAL ---
# Kelas ini kita salin langsung dari skrip inferensi sebelumnya.
# Ini adalah "otak" dari aplikasi kita.

class ModelRetrieval:
    def __init__(self, vectorizer_path, matrix_path, database_path):
        """
        Inisialisasi model dengan memuat semua komponen yang dibutuhkan.
        """
        try:
            print("Memuat model dan data...")
            if not all(os.path.exists(p) for p in [vectorizer_path, matrix_path, database_path]):
                raise FileNotFoundError("Satu atau lebih file model tidak ditemukan. Pastikan 'vectorizer.joblib', 'tfidf_matrix.joblib', dan 'retrieval_database.csv' ada.")
            
            self.vectorizer = joblib.load(vectorizer_path)
            self.tfidf_matrix = joblib.load(matrix_path)
            self.df_database = pd.read_csv(database_path)
            print("Model berhasil dimuat.")
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan fatal saat memuat model. Aplikasi tidak bisa berjalan. Error: {e}")
            self.vectorizer = None

    def _buat_kunci_emosi(self, input_emosi: dict):
        """
        Memproses input dictionary menjadi format kunci_emosi.
        """
        emosi_aktif = []
        for emosi, level in input_emosi.items():
            if level > 0:
                emosi_aktif.append(f"{emosi}_{int(level)}")
        emosi_aktif.sort()
        return " ".join(emosi_aktif)

    def cari_respons(self, input_emosi: dict):
        """
        Mencari dan mengembalikan insight, validation, dan saran yang paling cocok.
        """
        if self.vectorizer is None:
            return None

        kunci_input = self._buat_kunci_emosi(input_emosi)
        if not kunci_input:
            return None

        vector_input = self.vectorizer.transform([kunci_input])
        similarities = cosine_similarity(vector_input, self.tfidf_matrix)
        indeks_terbaik = similarities.argmax()
        skor_terbaik = similarities[0, indeks_terbaik]
        konten_terbaik = self.df_database.loc[indeks_terbaik, 'konten_gabungan']
        
        try:
            insight, validation, saran = konten_terbaik.split('|||')
            return {
                'insight': insight,
                'validation': validation,
                'saran': saran,
                'skor_kemiripan': skor_terbaik
            }
        except ValueError:
            return None

# --- LANGKAH 2: DEFINISIKAN STRUKTUR INPUT API ---
# Pydantic akan memvalidasi bahwa data yang masuk sesuai format ini.
class EmotionInput(BaseModel):
    anxiety: int = 0
    fear: int = 0
    nervousness: int = 0
    sadness: int = 0
    suffering: int = 0
    shame: int = 0

# --- LANGKAH 3: INISIALISASI APLIKASI FASTAPI DAN MODEL ---
# Buat instance aplikasi FastAPI
app = FastAPI(title="API Insight Journaling", version="1.0")

# Muat model saat aplikasi pertama kali dijalankan (startup)
# Ini memastikan model hanya dimuat sekali, bukan setiap kali ada permintaan.
print("Mempersiapkan aplikasi...")
model = ModelRetrieval(
    vectorizer_path='vectorizer.joblib',
    matrix_path='tfidf_matrix.joblib',
    database_path='retrieval_database.csv'
)

# --- LANGKAH 4: BUAT ENDPOINT API ---

@app.get("/", summary="Endpoint Cek Status", description="Endpoint dasar untuk memeriksa apakah API berjalan.")
def read_root():
    """
    Endpoint dasar untuk memeriksa apakah API berjalan.
    """
    return {"status": "API Insight Journaling berjalan."}


@app.post("/get-insight/", summary="Dapatkan Insight dari Emosi", description="Kirim data emosi dalam format JSON untuk mendapatkan insight, validasi, dan saran.")
def get_insight_api(emotions: EmotionInput):
    """
    Menerima input emosi dari pengguna, memprosesnya dengan model,
    dan mengembalikan hasil yang paling cocok.
    """
    if model.vectorizer is None:
        # Jika model gagal dimuat saat startup, kirim error 503
        raise HTTPException(status_code=503, detail="Model tidak tersedia atau gagal dimuat. Silakan cek log server.")

    # Ubah input dari Pydantic menjadi format dictionary
    input_dict = emotions.dict()
    
    # Panggil fungsi dari model kita untuk mencari respons
    hasil = model.cari_respons(input_dict)
    
    # Kirim hasilnya sebagai JSON
    if hasil:
        return hasil
    else:
        # Jika tidak ada hasil atau error, kirim error 404
        raise HTTPException(status_code=404, detail="Tidak dapat menemukan respons yang cocok untuk kombinasi emosi yang diberikan.")

# --- CARA MENJALANKAN APLIKASI INI ---
# 1. Pastikan Anda sudah menginstal library yang dibutuhkan:
#    pip install fastapi uvicorn python-multipart joblib scikit-learn pandas
#
# 2. Simpan kode ini dalam file bernama `main.py`.
#
# 3. Pastikan file `main.py`, `vectorizer.joblib`, `tfidf_matrix.joblib`, 
#    dan `retrieval_database.csv` berada di dalam folder yang sama.
#
# 4. Buka terminal Anda di folder tersebut, lalu jalankan perintah:
#    uvicorn main:app --reload
#
# 5. Buka browser Anda dan kunjungi http://127.0.0.1:8000/docs
#    Anda akan melihat dokumentasi API interaktif untuk mencoba endpoint ini.
