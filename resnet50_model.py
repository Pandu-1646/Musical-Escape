"""
ResNet-50 Model Architecture for GTZAN Music Genre Classification
Directly from the Research Paper:
"Music Genre Classification Using Deep Transfer Learning (ResNet-50)
An End-to-End Research Pipeline on the GTZAN Dataset"
Authors: B.V. Harsha Vardhan, B. Navadeep Reddy, M. Owaiz, P. Abhiram
"""

import torch
import torch.nn as nn
import torchvision.models as models

# The 10 GTZAN Dataset Genres
GENRE_NAMES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_NAMES)}
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRE_NAMES)}

def build_resnet50_model(num_classes=10, weights='DEFAULT'):
    """
    Constructs the fine-tuned ResNet-50 model with custom transfer learning head:
    Linear(2048 -> 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.4) -> Linear(512 -> 10)
    """
    if weights == 'DEFAULT' or weights is True:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)

    # Freeze backbone during feature extraction warmup
    for param in model.parameters():
        param.requires_grad = False

    # Specialized classification head for audio spectrograms
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_resnet50_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] ResNet-50 GTZAN Model initialized on {device}")
    print(f"[*] Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")
