import subprocess
import sys

# Install required libraries
print("Installing required libraries...")
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "librosa",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "pandas",
    "seaborn",
    "streamlit",
    "tqdm",
    "--quiet"
])
print("✓ Libraries installed!\n")

# ---- Check PyTorch + GPU ----
import torch
import numpy as np
import librosa
import time

print("=" * 50)
print("SYSTEM CHECK")
print("=" * 50)

print(f"PyTorch Version    : {torch.__version__}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Detected       : ✓ {gpu_name}")
    print(f"GPU Memory         : {gpu_memory:.1f} GB")
    device = torch.device("cuda")
else:
    print("GPU Detected       : ✗ No GPU - will use CPU (slower)")
    device = torch.device("cpu")

print(f"NumPy Version      : {np.__version__}")
print(f"Librosa Version    : {librosa.__version__}")
print(f"Device to use      : {device}")
print("=" * 50)

# ---- Quick GPU Speed Test ----
print("\nRunning quick GPU performance test...")

import torch.nn as nn

class TestCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TestCNN().to(device)
dummy_input = torch.randn(32, 1, 128, 128).to(device)

start = time.time()
with torch.no_grad():
    for _ in range(20):
        _ = model(dummy_input)
elapsed = time.time() - start

if device.type == 'cuda':
    torch.cuda.synchronize()

print(f"20 forward passes  : {elapsed:.2f} seconds")

if elapsed < 2:
    print("Performance        : ✓ EXCELLENT - GPU is working perfectly!")
elif elapsed < 5:
    print("Performance        : ✓ GOOD")
else:
    print("Performance        : ⚠ Running on CPU")

print("\n" + "=" * 50)
print("✓ SETUP COMPLETE - Ready to start!")
print("=" * 50)
print("\nNEXT STEP: Download GTZAN dataset from Kaggle")
print("https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
print("\nExtract it so your folder looks like:")
print("data/genres_original/blues/")
print("data/genres_original/classical/")
print("data/genres_original/... (10 folders)")
print("\nThen run: python step2_preprocess.py")
