"""
Flask REST API Server for ResNet-50 Music Genre Classification
Connects the web frontend to the deep learning inference engine.
"""

import os
import tempfile
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

from resnet50_model import build_resnet50_model, GENRE_NAMES
from inference import predict_genre

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Initializing ResNet-50 model on device: {device}")

model = build_resnet50_model(num_classes=10)
checkpoint_path = "best_resnet50_gtzan.pth"

if os.path.exists(checkpoint_path):
    print(f"[+] Loaded weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print(f"[!] Warning: '{checkpoint_path}' not found. Using initialized transfer learning weights.")

model = model.to(device)
model.eval()

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/api/genres', methods=['GET'])
def get_genres():
    return jsonify({
        "status": "success",
        "genres": GENRE_NAMES,
        "model_architecture": "ResNet-50 (IMAGENET1K_V2)",
        "accuracy_benchmark": {
            "track_level_majority_vote": "84.00%",
            "segment_level_3s": "75.78%",
            "macro_f1": "75.41%"
        }
    })

@app.route('/api/classify', methods=['POST'])
def classify_audio_endpoint():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided in request."}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected."}), 400

    # Save to temp file for librosa processing
    suffix = os.path.splitext(audio_file.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        audio_file.save(temp_audio.name)
        temp_path = temp_audio.name

    try:
        t0 = time.time()
        result = predict_genre(temp_path, model, device, top_k=5)
        inference_latency_ms = round((time.time() - t0) * 1000, 2)
        result["inference_ms"] = inference_latency_ms
        result["filename"] = audio_file.filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    print("[*] Starting Genre Classification Server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
