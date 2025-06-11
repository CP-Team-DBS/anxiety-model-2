import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from scipy.sparse import save_npz # Untuk menyimpan matriks

def buat_dan_simpan_model_robust(file_database, file_vectorizer_config, file_matrix):
    """
    Fungsi untuk membuat model TF-IDF dan menyimpannya dalam format
    JSON dan NPZ yang lebih andal.
    """
    try:
        print(f"Membaca database dari '{file_database}'...")
        df = pd.read_csv(file_database)
        print("Database berhasil dibaca.")
    except FileNotFoundError:
        print(f"ERROR: File database '{file_database}' tidak ditemukan.")
        return

    # PERBAIKAN: Mengatasi FutureWarning dari pandas
    df['kunci_emosi'] = df['kunci_emosi'].fillna('')
    kunci_emosi_list = df['kunci_emosi'].tolist()
    
    print("\nMembuat TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(kunci_emosi_list)
    print("Vektorisasi TF-IDF selesai.")
    
    try:
        # PERBAIKAN: Mengonversi parameter yang tidak bisa diserialisasi
        params = vectorizer.get_params()
        for key, value in params.items():
            if isinstance(value, type):
                params[key] = value.__name__  # Ubah tipe data (e.g., <class 'numpy.float64'>) menjadi string "float64"

        vectorizer_config = {
            'vocabulary_': vectorizer.vocabulary_,
            'idf_': vectorizer.idf_.tolist(),
            'params': params # Gunakan parameter yang sudah dibersihkan
        }
        
        with open(file_vectorizer_config, 'w', encoding='utf-8') as f:
            json.dump(vectorizer_config, f)
        print(f"Konfigurasi vectorizer disimpan ke '{file_vectorizer_config}'")
        
        save_npz(file_matrix, tfidf_matrix)
        print(f"Matriks TF-IDF disimpan ke '{file_matrix}'")
        
        print("\nBERHASIL! Aset model retrieval telah disimpan dalam format baru.")
    except Exception as e:
        print(f"\nERROR: Gagal menyimpan model. Error: {e}")

if __name__ == "__main__":
    database_csv = 'retrieval_database.csv'
    vectorizer_path = 'api/vectorizer_config.json'
    matrix_path = 'api/tfidf_matrix.npz'
    
    buat_dan_simpan_model_robust(database_csv, vectorizer_path, matrix_path)
