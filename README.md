# Musical Escape — Music Genre Classification (GTZAN Dataset)

An interactive deep-learning music genre classification web application and PyTorch ResNet-50 pipeline built around the GTZAN 10-genre standard (*Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock*).

---

## 🚀 Key Features

1. **Interactive Glassmorphism Web Application ([`index.html`](index.html))**:
   - **Real-Time Browser Deep Learning (`AcousticResNet`)**: Multi-segment Web Audio API decoding, 26-band Mel-filterbank & 13 MFCC extraction, Parseval RMS band ratios, spectral flux, and a deterministic 2-block Residual Neural Network with segment attention pooling and majority voting.
   - **Live Playback & Visualizers**: Interactive waveform scrubber (Frequency Bars, Waveform Line, Mirror Bars), play/pause/prev/next/shuffle/loop controls, and real-time FFT volume control.
   - **Analysis & Uploads Views**: 2D Log-Mel Spectrogram heatmap canvas, live frequency spectrum, extracted acoustic features, and uploaded track manager.
   - **Pure CSS & Vector UI**: Self-contained tropical dusk visual theme with zero external image dependencies.

2. **PyTorch ResNet-50 Model ([`resnet50_model.py`](resnet50_model.py))**:
   - ResNet-50 transfer learning backbone (`IMAGENET1K_V2`) with custom classification head for the 10 GTZAN genres.

3. **Inference Engine ([`inference.py`](inference.py)) & Flask API ([`server.py`](server.py))**:
   - Segments audio into 3-second slices, extracts 128-bin Log-Mel Spectrograms (`sr=22,050Hz`, `n_fft=2048`, `hop=512`), and performs track-level majority voting via `POST /api/classify`.

4. **Zero-Dependency Local Server ([`run_server.ps1`](run_server.ps1))**:
   - Lightweight Windows PowerShell `HttpListener` server (`http://localhost:8181/`) for running the standalone web interface without Python or Node.js.

---

## 📂 Project Structure

```text
genre_classifier/
├── index.html          # Self-contained frontend & client-side AcousticResNet classifier
├── run_server.ps1      # Standalone PowerShell local HTTP server (port 8181)
├── resnet50_model.py   # PyTorch ResNet-50 transfer learning architecture
├── inference.py        # CLI & segment-level PyTorch inference engine
├── server.py           # Flask REST API backend
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

---

## 🛠️ Quick Start

### Option A: Run Web Interface with PowerShell (No Dependencies)
```powershell
powershell -ExecutionPolicy Bypass -File .\run_server.ps1
```
Then open **`http://localhost:8181/`** in Chrome.

### Option B: Run Full Python / PyTorch Backend
```bash
pip install torch torchvision librosa flask flask-cors
python server.py
```
Then open **`http://localhost:5000/`**.

### Option C: CLI Inference
```bash
python inference.py --audio my_audio_track.wav --weights best_resnet50_gtzan.pth
```
