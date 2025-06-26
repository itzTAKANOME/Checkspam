import numpy as np
import joblib
import re
import os
import requests
from flask import Flask, request, render_template_string, jsonify
from googletrans import Translator

# Inisialisasi aplikasi Flask
app = Flask(__name__)
translator = Translator()

# --- HTML Template dalam satu string ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detektor Spam/Phishing</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; color: #333; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px 0; }
        .container { background-color: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1); width: 100%; max-width: 600px; text-align: center; }
        h1 { color: #1d2129; margin-bottom: 20px; }
        textarea { width: 100%; padding: 15px; border-radius: 8px; border: 1px solid #dddfe2; margin-bottom: 20px; font-size: 16px; box-sizing: border-box; min-height: 150px; resize: vertical; }
        button { background-color: #007bff; color: white; padding: 15px 25px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; transition: background-color 0.3s; }
        button:hover { background-color: #0056b3; }
        .result-container { margin-top: 20px; }
        .translation-info { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: left; }
        .translation-info p { margin: 5px 0; }
        .translation-info strong { color: #495057; }
        .prediction { padding: 20px; border-radius: 8px; font-size: 1.2em; font-weight: bold; }
        .prediction.spam { background-color: #ffebee; color: #c62828; }
        .prediction.ham { background-color: #e8f5e9; color: #2e7d32; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cek Pesan Spam atau Ham</h1>
        <form action="/predict" method="post">
            <textarea name="text" placeholder="Masukkan teks pesan atau email di sini..." required>{{ text_input or '' }}</textarea>
            <br>
            <button type="submit">Prediksi</button>
        </form>
        
        {% if prediction_text %}
            <div class="result-container">
                {% if translated_text %}
                    <div class="translation-info">
                        <p><strong>Bahasa Terdeteksi:</strong> {{ detected_lang }}</p>
                        <p><strong>Teks Diterjemahkan (untuk Analisis):</strong><br>{{ translated_text }}</p>
                    </div>
                {% endif %}
                <div class="prediction {% if 'SPAM' in prediction_text %}spam{% else %}ham{% endif %}">
                    {{ prediction_text }}
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# --- Fungsi-fungsi untuk Aplikasi ---

def download_file_from_url(url, local_path):
    """Mengunduh file dari URL jika belum ada di path lokal."""
    if not os.path.exists(local_path):
        print(f"Mengunduh file dari {url} ke {local_path}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Unduhan selesai.")
    else:
        print(f"File {local_path} sudah ada.")
    return local_path

def get_glove_vector(sentence, embeddings_index, embedding_dim=300):
    words = sentence.split()
    word_vectors = []
    for word in words:
        vector = embeddings_index.get(word)
        if vector is not None:
            word_vectors.append(vector)
    if len(word_vectors) == 0:
        return np.zeros(embedding_dim)
    return np.mean(word_vectors, axis=0)

def transform_single_text_to_glove(text, embeddings_index, embedding_dim=300):
    vector = get_glove_vector(text, embeddings_index, embedding_dim)
    return np.array([vector])

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# --- Memuat Model dan Embeddings ---

# Vercel menyediakan direktori /tmp yang bisa ditulis
TMP_DIR = '/tmp'
MODEL_PATH = os.path.join(TMP_DIR, 'random_forest_model_glove.pkl')
EMBEDDINGS_PATH = os.path.join(TMP_DIR, 'glove_embeddings.pkl')

MODEL_URL = "https://drive.google.com/file/d/10fIPl0nJrdpWU460AxSGRre1bb-7UzGk/view?usp=drive_link"
EMBEDDINGS_URL = "https://drive.google.com/file/d/1V3XdjTd0pNkPKX9eTJFmLhZDSF6Mp09V/view?usp=drive_link"

model = None
embeddings_index = None

try:
    # Unduh dan muat model
    print("Memproses model...")
    download_file_from_url(MODEL_URL, MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    print(">>> SUKSES: Model berhasil dimuat.")

    # Unduh dan muat embeddings
    print("\nMemproses embeddings...")
    download_file_from_url(EMBEDDINGS_URL, EMBEDDINGS_PATH)
    embeddings_index = joblib.load(EMBEDDINGS_PATH)
    print(">>> SUKSES: Embeddings berhasil dimuat.")

except Exception as e:
    print(f">>> ERROR saat inisialisasi: {e}")

if model is None or embeddings_index is None:
    print("\nAPLIKASI TIDAK DAPAT DIMULAI karena model atau embeddings tidak berhasil dimuat.")
else:
    print("\nModel dan embeddings siap.")


# --- Rute Aplikasi Web ---

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # Rute ini menangani semua permintaan
    if request.method == 'POST' and path == 'predict':
        return predict()
    elif request.method == 'GET':
        return home()
    else:
        return "Not Found", 404

def home():
    """Menampilkan halaman utama."""
    return render_template_string(HTML_TEMPLATE)

def predict():
    """Menerima input, melakukan prediksi, dan mengembalikan respons."""
    if model is None or embeddings_index is None:
        error_msg = {"error": "Model atau embeddings tidak siap."}
        return jsonify(error_msg) if request.is_json else render_template_string(HTML_TEMPLATE, prediction_text=error_msg['error'])

    is_api_request = request.is_json
    
    if is_api_request:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Payload JSON tidak valid. Harus mengandung key 'text'."}), 400
        text_input = data.get('text', '')
    else:
        text_input = request.form.get('text', '')
    
    if not text_input:
        error_msg = {"error": "Input teks tidak boleh kosong."}
        return jsonify(error_msg) if is_api_request else render_template_string(HTML_TEMPLATE)
        
    translated_text = None
    detected_lang = 'en'

    try:
        detection = translator.detect(text_input)
        detected_lang = detection.lang
        text_to_process = translator.translate(text_input, dest='en').text if detected_lang != 'en' else text_input
        if detected_lang != 'en':
            translated_text = text_to_process
    except Exception as e:
        print(f"Error saat menerjemahkan: {e}")
        text_to_process = text_input
        
    processed_text = preprocess_text(text_to_process)
    vectorized_text = transform_single_text_to_glove(processed_text, embeddings_index, embedding_dim=300)
    prediction = model.predict(vectorized_text)
    prediction_proba = model.predict_proba(vectorized_text)
    
    result = prediction[0]
    class_index = list(model.classes_).index(result)
    confidence = prediction_proba[0][class_index]
    
    if is_api_request:
        return jsonify({
            "prediction": result.upper(),
            "confidence": f"{confidence:.2%}",
            "original_text": text_input,
            "detected_language": detected_lang,
            "processed_text": translated_text or text_input
        })
    else:
        output_text = f"Prediksi: {result.upper()} (Kepercayaan: {confidence:.2%})"
        return render_template_string(HTML_TEMPLATE, 
                               prediction_text=output_text, 
                               text_input=text_input,
                               translated_text=translated_text,
                               detected_lang=detected_lang)
