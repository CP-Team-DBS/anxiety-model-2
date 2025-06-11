# api_model_1.py
# Impor semua library yang dibutuhkan
import os
import re
import string
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deep_translator import GoogleTranslator
import json

# --- BAGIAN 1: FUNGSI PEMBANTU ---
# Fungsi-fungsi ini diadaptasi dari notebook pelatihan Anda.

def preprocess_text(text):
    """
    Fungsi untuk membersihkan teks: mengubah ke huruf kecil, 
    menghapus tanda baca, dan angka.
    """
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

def top_3_level_emotions(prediction_dict, all_emotion_labels):
    """
    PERBAIKAN: Mengambil 3 emosi dengan skor tertinggi, mengubahnya 
    menjadi level (0-3), dan menyetel emosi lainnya ke level 0.
    Ini meniru logika dari fungsi `top_3_level_emotions` di notebook Anda.
    """
    def score_to_level(score):
        # Logika konversi skor ke level dari notebook Anda
        if score == 0:
            return 0
        elif score <= 0.05:
            return 1
        elif score <= 0.25:
            return 2
        else:
            return 3

    # Urutkan emosi berdasarkan skor dari yang tertinggi ke terendah
    sorted_emotions_by_score = sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Dapatkan nama dari 3 emosi teratas
    top_3_emotion_names = [emotion for emotion, score in sorted_emotions_by_score[:3]]
    
    # Inisialisasi kamus hasil dengan semua emosi bernilai 0
    final_emotion_levels = {label: 0 for label in all_emotion_labels}

    # Hitung dan tetapkan level hanya untuk 3 emosi teratas
    for emotion, score in sorted_emotions_by_score[:3]:
        if score > 0: # Hanya konversi jika skornya positif
             final_emotion_levels[emotion] = score_to_level(score)
            
    return final_emotion_levels

# --- BAGIAN 2: PENGATURAN APLIKASI FASTAPI ---

# Definisikan struktur input untuk API
class TextInput(BaseModel):
    journal_text: str

# Buat instance aplikasi
app = FastAPI(title="API Deteksi Emosi (Model 1)", version="1.0")

# Tempat untuk menyimpan aset model yang sudah dimuat
model_assets = {}

@app.on_event("startup")
def load_model_assets():
    """
    Memuat semua aset Model 1 saat aplikasi pertama kali dijalankan.
    """
    print("Memuat aset Model 1...")
    # Dapatkan path direktori tempat skrip ini berada
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Tentukan path ke file aset
        model_path = os.path.join(base_dir, 'emotion_regression_model_v2.tflite')
        tokenizer_path = os.path.join(base_dir, 'tokenizer_config.json')

        # Muat interpreter TFLite
        model_assets['interpreter'] = tf.lite.Interpreter(model_path=model_path)
        model_assets['interpreter'].allocate_tensors()
        
        # Muat tokenizer dari file JSON
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_config = f.read()
        model_assets['tokenizer'] = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

        # Simpan parameter lain
        model_assets['maxlen'] = 100
        model_assets['emotion_labels'] = ['anxiety', 'fear', 'nervousness', 'sadness', 'suffering', 'shame']
        
        print("Aset Model 1 berhasil dimuat.")
    except Exception as e:
        print(f"ERROR FATAL: Gagal memuat aset. Aplikasi mungkin tidak berfungsi. Error: {e}")
        model_assets.clear()

# --- BAGIAN 3: ENDPOINT API ---

@app.get("/", summary="Cek Status API")
def read_root():
    return {"status": "API Deteksi Emosi berjalan."}

@app.post("/predict-emotion/", summary="Prediksi Emosi dari Teks")
def predict_emotion(data: TextInput):
    """
    Menerima teks jurnal, menerjemahkannya, melakukan pra-pemrosesan, 
    dan mengembalikan prediksi level emosi dari Model 1.
    """
    if not model_assets:
        raise HTTPException(status_code=503, detail="Model tidak tersedia karena gagal dimuat saat startup.")

    try:
        # 1. Terjemahkan teks
        translated_text = GoogleTranslator(source='auto', target='en').translate(data.journal_text)
        
        # 2. Pra-proses, tokenisasi, dan padding
        processed_text = preprocess_text(translated_text)
        seq = model_assets['tokenizer'].texts_to_sequences([processed_text])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=model_assets['maxlen'], padding='post', truncating='post')
        input_data = np.array(padded_seq, dtype=np.float32)

        # 3. Prediksi dengan TFLite
        interpreter = model_assets['interpreter']
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        # 4. Konversi skor menjadi level emosi
        emotion_scores = {label: float(score) for label, score in zip(model_assets['emotion_labels'], raw_prediction)}
        # Panggil fungsi yang sudah diperbarui untuk logika top 3
        predicted_emotions = top_3_level_emotions(emotion_scores, model_assets['emotion_labels'])

        # 5. Kembalikan hasil
        return {
            "input_text": data.journal_text,
            "translated_text": translated_text,
            "predicted_emotions": predicted_emotions,
            "raw_scores": emotion_scores
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat pemrosesan. Error: {e}")

# --- CARA MENJALANKAN ---
# 1. Pastikan folder `emotion_detection_model` berisi file ini (`api_model_1.py`), 
#    `emotion_regression_model.tflite`, dan `tokenizer_config.json`.
# 2. Instal library dari file `requirements.txt` Anda.
# 3. Dari terminal di folder root proyek (`C:\DBS\capstone_project_ml`), jalankan:
#    uvicorn emotion_detection_model.api_model_1:app --reload
# 4. Buka browser ke http://127.0.0.1:8000/docs untuk mencoba.
