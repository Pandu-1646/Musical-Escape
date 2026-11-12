"""
Inference Engine for GTZAN Music Genre Classification using ResNet-50
Directly from Section 8 of the Research Paper:
"Inference Engine: Testing Custom / Unseen Audio Files"
"""

import os
import sys
import math
import argparse
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from resnet50_model import build_resnet50_model, GENRE_NAMES, IDX_TO_GENRE

# Audio feature extraction constants matching the paper protocol
SAMPLE_RATE = 22050
DURATION = 3.0
SAMPLES_PER_SEGMENT = int(SAMPLE_RATE * DURATION)  # 66,150 samples per slice
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

def extract_mel_spectrogram(chunk, sr=SAMPLE_RATE):
    """
    Computes normalized Log-Mel Spectrogram for a 3-second audio slice.
    """
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Min-max normalization into [0, 1]
    denom = log_mel.max() - log_mel.min()
    norm_spec = (log_mel - log_mel.min()) / (denom if denom != 0 else 1.0)
    return norm_spec.astype(np.float32)

def predict_genre(audio_path, model, device, top_k=5):
    """
    Predicts the music genre of any arbitrary audio file (.wav, .mp3, etc.)
    using track-level majority voting and averaged segment probabilities.
    """
    model.eval()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load and resample audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"[*] Processing Audio: '{os.path.basename(audio_path)}' (Duration: {duration:.2f}s, SR: {sr}Hz)")

    # 2. Slice into 3-second non-overlapping chunks
    num_slices = math.floor(len(y) / SAMPLES_PER_SEGMENT)
    if num_slices == 0:
        # If track is shorter than 3 seconds, zero-pad
        y = np.pad(y, (0, SAMPLES_PER_SEGMENT - len(y)))
        num_slices = 1

    segment_probs = []
    segment_votes = []
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    with torch.no_grad():
        for s in range(num_slices):
            start = s * SAMPLES_PER_SEGMENT
            end = start + SAMPLES_PER_SEGMENT
            chunk = y[start:end]

            # Compute Log-Mel Spectrogram
            norm_spec = extract_mel_spectrogram(chunk, sr=sr)

            # PyTorch Tensor transformation
            t = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
            # Bilinear interpolation to (224, 224) matching ResNet-50 ImageNet input
            t_resized = F.interpolate(t.unsqueeze(0), size=(224, 224), mode='bilinear')
            t_3ch = t_resized.repeat(1, 3, 1, 1)  # Expand to 3 identical RGB channels
            t_norm = normalize(t_3ch.squeeze(0)).unsqueeze(0).to(device)

            output = model(t_norm)
            prob = F.softmax(output, dim=1).cpu().numpy()[0]
            segment_probs.append(prob)
            segment_votes.append(int(prob.argmax()))

    # Track-level average probabilities across all segments
    avg_probs = np.mean(segment_probs, axis=0)
    top_indices = np.argsort(avg_probs)[::-1][:top_k]

    predicted_genre = IDX_TO_GENRE[top_indices[0]]
    predicted_confidence = float(avg_probs[top_indices[0]] * 100)

    # Majority vote across segments
    majority_vote_genre = IDX_TO_GENRE[max(set(segment_votes), key=segment_votes.count)]

    print(f"\n=======================================================")
    print(f"🎵 ResNet-50 GTZAN Prediction Results")
    print(f"=======================================================")
    print(f"[*] Top Predicted Genre : {predicted_genre.upper()} ({predicted_confidence:.2f}%)")
    print(f"[*] Majority Vote Genre  : {majority_vote_genre.upper()} ({segment_votes.count(top_indices[0])}/{num_slices} segments)")
    print(f"[*] Total Slices Evaluated: {num_slices} segments (3.0s each)")
    print(f"\n--- Top {top_k} Genre Probabilities ---")
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        genre_name = IDX_TO_GENRE[idx]
        conf = float(avg_probs[idx] * 100)
        results.append({"rank": rank, "genre": genre_name, "confidence": conf})
        print(f"  {rank}. {genre_name:<10}: {conf:6.2f}%")

    return {
        "predicted_genre": predicted_genre,
        "confidence": predicted_confidence,
        "majority_genre": majority_vote_genre,
        "num_slices": num_slices,
        "top_predictions": results,
        "all_probabilities": {IDX_TO_GENRE[i]: float(avg_probs[i] * 100) for i in range(len(GENRE_NAMES))}
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify music genre using ResNet-50 on GTZAN")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (.wav or .mp3)")
    parser.add_argument("--weights", type=str, default="best_resnet50_gtzan.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Target Device: {device}")

    model = build_resnet50_model(num_classes=10)
    if os.path.exists(args.weights):
        print(f"[*] Loading checkpoint weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print(f"[!] Checkpoint '{args.weights}' not found. Running with ImageNet-initialized weights.")

    model = model.to(device)
    predict_genre(args.audio, model, device)
