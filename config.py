import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
PROFILES_DIR = BASE_DIR / "profiles"

# Ensure directories exist
for p in [DATA_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR, PROFILES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Audio Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5   # Window duration in seconds for real-time inference
STEP_DURATION = 0.5    # Stride / step duration for rolling inference
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 96     # Fixed time frames for 1.5s window @ 16kHz with hop 256

# Risk Scoring Weights
WEIGHT_ACOUSTIC = 0.45    # Spectral synthesis / vocoder artifact probability
WEIGHT_PROSODY = 0.25     # Robotic / unnatural pitch and pause metrics
WEIGHT_IDENTITY = 0.30    # Target speaker biometric mismatch

# Threat Level Thresholds
THRESHOLD_SAFE = 0.35
THRESHOLD_SUSPICIOUS = 0.70
THRESHOLD_CRITICAL = 0.70

# Hardware
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
