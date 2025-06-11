# main_api.py
import os
import re
import string
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator
from scipy.sparse import load_npz

# --- BAGIAN 1: FUNGSI PEMBANTU UNTUK MODEL 1 ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

def top_3_level_emotions(prediction_dict, all_emotion_labels):
    """Mengambil 3 emosi teratas dan mengubahnya ke level 0-3."""
    def score_to_level(score):
        if score == 0: return 0
        elif score <= 0.05: return 1
        elif score <= 0.25: return 2
        else: return 3

    sorted_emotions = sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True)
    final_levels = {label: 0 for label in all_emotion_labels}

    for emotion, score in sorted_emotions[:3]:
        if score > 0:
            final_levels[emotion] = score_to_level(score)
            
    return final_levels

# --- BAGIAN 2: KELAS UNTUK MODEL 2 ---
class ModelRetrieval:
    def __init__(self, vectorizer, tfidf_matrix, df_database):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.df_database = df_database

    def _buat_kunci_emosi(self, input_emosi: dict):
        emosi_aktif = [f"{emosi}_{int(level)}" for emosi, level in input_emosi.items() if level > 0]
        emosi_aktif.sort()
        return " ".join(emosi_aktif)

    def cari_respons(self, input_emosi: dict):
        kunci_input = self._buat_kunci_emosi(input_emosi)
        if not kunci_input: return None

        vector_input = self.vectorizer.transform([kunci_input])
        similarities = cosine_similarity(vector_input, self.tfidf_matrix)
        indeks_terbaik = similarities.argmax()
        
        konten_terbaik = self.df_database.loc[indeks_terbaik, 'konten_gabungan']
        try:
            insight, validation, saran = konten_terbaik.split('|||')
            return {'insight': insight, 'validation': validation, 'saran': saran}
        except ValueError:
            return None

# --- BAGIAN 3: PENGATURAN APLIKASI FASTAPI ---
class TextInput(BaseModel):
    journal_text: str

app = FastAPI(title="API Analisis Journaling Lengkap", version="3.0")
models = {}

@app.on_event("startup")
def load_models():
    """Memuat semua aset saat aplikasi dijalankan."""
    print("Memuat semua model dan aset...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Aset Model 1
        model1_path = os.path.join(base_dir, 'emotion_regression_model_v2.tflite')
        tokenizer_path = os.path.join(base_dir, 'tokenizer_config.json')

        models['interpreter'] = tf.lite.Interpreter(model_path=model1_path)
        models['interpreter'].allocate_tensors()
        
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            models['tokenizer'] = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

        models['maxlen'] = 100
        models['emotion_labels'] = ['anxiety', 'fear', 'nervousness', 'sadness', 'suffering', 'shame']

        # Aset Model 2
        vectorizer_path = os.path.join(base_dir, 'vectorizer_config.json')
        matrix_path = os.path.join(base_dir, 'tfidf_matrix.npz')
        database_path = os.path.join(base_dir, 'retrieval_database.csv')

        with open(vectorizer_path, 'r', encoding='utf-8') as f:
            vectorizer_config = json.load(f)
        
        vectorizer = TfidfVectorizer(**vectorizer_config['params'])
        vectorizer.vocabulary_ = vectorizer_config['vocabulary_']
        vectorizer.idf_ = np.array(vectorizer_config['idf_'])
        
        tfidf_matrix = load_npz(matrix_path)
        df_database = pd.read_csv(database_path)
        models['retrieval_model'] = ModelRetrieval(vectorizer, tfidf_matrix, df_database)
        
        print("Semua model berhasil dimuat.")
    except Exception as e:
        print(f"ERROR FATAL: Gagal memuat salah satu model. Aplikasi mungkin tidak berfungsi. Error: {e}")
        models.clear()

# --- BAGIAN 4: ENDPOINT API UTAMA ---
@app.get("/")
def read_root():
    return {"status": "API Analisis Journaling berjalan."}

@app.post("/process-journal/", summary="Proses Teks Jurnal untuk Dapatkan Insight")
def process_journal_entry(data: TextInput):
    if not models:
        raise HTTPException(status_code=503, detail="Model tidak tersedia karena gagal dimuat.")

    # --- Jalankan Model 1 ---
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(data.journal_text)
        processed_text = preprocess_text(translated_text)
        seq = models['tokenizer'].texts_to_sequences([processed_text])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=models['maxlen'], padding='post')
        input_data = np.array(padded_seq, dtype=np.float32)

        interpreter = models['interpreter']
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        emotion_scores = {label: float(score) for label, score in zip(models['emotion_labels'], raw_prediction)}
        predicted_emotions = top_3_level_emotions(emotion_scores, models['emotion_labels'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal saat Deteksi Emosi (Model 1). Error: {e}")

    # --- Jalankan Model 2 ---
    try:
        retrieval_model = models['retrieval_model']
        retrieved_content = retrieval_model.cari_respons(predicted_emotions)

        if retrieved_content:
            # Format respons sesuai permintaan Anda
            return {
                "predicted_emotions": predicted_emotions,
                "insight": retrieved_content['insight'],
                "validation": retrieved_content['validation'],
                "saran": retrieved_content['saran']
            }
        else:
            raise HTTPException(status_code=404, detail="Tidak dapat menemukan respons yang cocok.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal saat Retrieval (Model 2). Error: {e}")

# --- CARA MENJALANKAN ---
# 1. Jalankan `buat_tokenizer.py` dan `buat_model_retrieval.py` terlebih dahulu.
# 2. Pastikan folder `api` berisi semua file aset yang diperlukan.
# 3. Jalankan server dari folder root proyek: uvicorn api.main_api:app --reload
