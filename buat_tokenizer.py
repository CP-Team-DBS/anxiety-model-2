import pandas as pd
import re
import string
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions

# --- FUNGSI PREPROCESSING DARI NOTEBOOK ANDA ---
def setup_nltk():
    """Unduh resource NLTK yang diperlukan."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Mengunduh stopwords NLTK...")
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Mengunduh wordnet NLTK...")
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("Mengunduh omw-1.4 NLTK...")
        nltk.download('omw-1.4')

def clean_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, stop_words, lemmatizer):
    text = clean_text(text)
    words = text.split()
    processed_words = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemma = lemmatizer.lemmatize(lemma, pos='n')
            lemma = lemmatizer.lemmatize(lemma, pos='a')
            lemma = lemmatizer.lemmatize(lemma, pos='r')
            processed_words.append(lemma)
    return ' '.join(processed_words)

# --- SKRIP UTAMA ---
if __name__ == "__main__":
    print("Memulai proses pembuatan tokenizer config...")
    setup_nltk()
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'like', 'get', 'go', 'know', 'would', 'could', 'also'}
    stop_words.update(custom_stopwords)
    lemmatizer = WordNetLemmatizer()

    try:
        df = pd.read_csv("dataset.csv")
        print("Dataset 'dataset.csv' berhasil dibaca.")
    except FileNotFoundError:
        print("ERROR: Pastikan file 'dataset.csv' berada di folder root proyek Anda.")
        exit()

    print("Melakukan preprocessing teks pada kolom 'statement'...")
    df['statement'] = df['statement'].astype(str)
    X = df['statement'].apply(lambda text: preprocess_text(text, stop_words, lemmatizer))

    print("Membuat dan melatih tokenizer...")
    vocab_size = 10000
    oov_token = '<OOV>'
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X)
    print("Tokenizer berhasil dilatih.")

    # Simpan konfigurasi tokenizer sebagai file JSON
    tokenizer_config = tokenizer.to_json()
    file_path = 'tokenizer_config.json'
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(tokenizer_config)
            
        print(f"\nBERHASIL! Konfigurasi Tokenizer telah disimpan sebagai '{file_path}'.")
        print("Sekarang, pindahkan file ini ke dalam folder 'api' Anda.")
    except Exception as e:
        print(f"\nERROR: Gagal menyimpan tokenizer config. Error: {e}")
