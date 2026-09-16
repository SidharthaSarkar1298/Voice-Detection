import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from config import MODELS_DIR, DATA_DIR, DEVICE, TARGET_FRAMES, SAMPLE_RATE
from core.feature_extractor import FeatureExtractor
from core.acoustic_model import VoiceShieldAcousticCNN

def apply_acoustic_channel_simulation(wav_np, sr=16000):
    """
    Simulates real-world phone speaker transmission & room acoustics:
    - Mobile phone speaker bandpass roll-off (250 Hz - 4500 Hz)
    - Early room reflections (reverb)
    - Low-level room noise
    """
    nyq = sr / 2.0
    low = 250.0 / nyq
    high = min(0.95, 4500.0 / nyq)
    b, a = signal.butter(4, [low, high], btype="band")
    filtered = signal.lfilter(b, a, wav_np)

    # 25ms early reflection
    delay = int(sr * 0.025)
    reverb = np.copy(filtered)
    if len(reverb) > delay:
        reverb[delay:] += 0.22 * filtered[:-delay]

    noise = np.random.normal(0, 0.005, len(reverb)).astype(np.float32)
    augmented = reverb + noise

    peak = np.max(np.abs(augmented))
    if peak > 1e-4:
        augmented = augmented / peak * 0.90
    return augmented.astype(np.float32)

class UnifiedVoiceShieldDataset(Dataset):
    def __init__(self, file_list, labels, feature_extractor, is_train=True):
        self.files = file_list
        self.labels = labels
        self.fe = feature_extractor
        self.is_train = is_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        try:
            wav, sr = sf.read(str(path))
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)

            # Audio domain channel simulation (for 40% of training samples)
            if self.is_train and np.random.rand() < 0.40:
                wav = apply_acoustic_channel_simulation(wav, sr=sr if sr else SAMPLE_RATE)

            wav_tensor = self.fe.preprocess_audio(wav, sr)
            mel = self.fe.extract_mel_spectrogram(wav_tensor)

            # SpecAugment on spectrogram during training
            if self.is_train:
                # Time mask
                if np.random.rand() < 0.35:
                    t_len = np.random.randint(4, 12)
                    t_start = np.random.randint(0, max(1, mel.shape[1] - t_len))
                    mel[:, t_start:t_start + t_len] = -80.0
                # Frequency mask
                if np.random.rand() < 0.35:
                    f_len = np.random.randint(4, 10)
                    f_start = np.random.randint(0, max(1, mel.shape[0] - f_len))
                    mel[f_start:f_start + f_len, :] = -80.0
                # Random Gaussian noise
                if np.random.rand() < 0.25:
                    mel = mel + np.random.normal(0, 0.35, mel.shape).astype(np.float32)

            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0) # (1, 80, 96)
            y = torch.tensor([label], dtype=torch.float32)
            return x, y
        except Exception as e:
            # Fallback zero tensor
            x = torch.zeros((1, 80, TARGET_FRAMES), dtype=torch.float32)
            y = torch.tensor([label], dtype=torch.float32)
            return x, y

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return round(float(eer) * 100, 2)

def train_unified_model():
    print("=" * 65)
    print("SIH26104 VOICE SHIELD: UNIFIED TRAINING PIPELINE")
    print(f"Target Device: {DEVICE.upper()} (GPU Accelerated)")
    print("=" * 65)

    test_dir = root_dir / "test"
    
    # 1. Collect Real Audio Files (IndicVoices + Baseline)
    real_files = []
    indic_real_dir = DATA_DIR / "indic_real"
    if indic_real_dir.exists():
        real_files.extend(list(indic_real_dir.glob("*.wav")))
    data_real_dir = DATA_DIR / "real"
    if data_real_dir.exists():
        real_files.extend(list(data_real_dir.glob("*.wav")))
    test_real_dir = test_dir / "real"
    if test_real_dir.exists():
        real_files.extend(list(test_real_dir.glob("*.wav")))

    # 2. Collect Fake Audio Files (WaveFake + ElevenLabs + Edge-TTS)
    fake_files = []
    wf_dir = DATA_DIR / "wavefake_fake"
    if wf_dir.exists():
        fake_files.extend(list(wf_dir.glob("*.wav")))
    data_fake_dir = DATA_DIR / "fake"
    if data_fake_dir.exists():
        fake_files.extend(list(data_fake_dir.glob("*.wav")))
    eleven_dir = test_dir / "Fake Voices 11 Labs"
    if eleven_dir.exists():
        fake_files.extend(list(eleven_dir.glob("*.wav")))
    test_fake_dir = test_dir / "fake"
    if test_fake_dir.exists():
        fake_files.extend(list(test_fake_dir.glob("*.wav")))

    print(f"Found {len(real_files)} Real files (IndicVoices & Multi-formant)")
    print(f"Found {len(fake_files)} Fake files (WaveFake, ElevenLabs, Edge-TTS)")

    # Balance datasets
    min_count = min(len(real_files), len(fake_files))
    np.random.seed(42)
    np.random.shuffle(real_files)
    np.random.shuffle(fake_files)
    
    real_files = real_files[:min_count]
    fake_files = fake_files[:min_count]

    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    print(f"Balanced Dataset: {len(all_files)} total files ({len(real_files)} Real / {len(fake_files)} Fake)")

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.20, random_state=42, stratify=all_labels
    )

    fe = FeatureExtractor()
    train_ds = UnifiedVoiceShieldDataset(train_files, train_labels, fe, is_train=True)
    val_ds = UnifiedVoiceShieldDataset(val_files, val_labels, fe, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = VoiceShieldAcousticCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-5)

    best_val_f1 = 0.0
    best_weights_path = MODELS_DIR / "voice_shield_cnn.pt"

    print("\nStarting GPU Training across 16 Epochs...")
    print("-" * 65)

    for epoch in range(1, 17):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_preds, y_probs = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)

                y_probs.extend(probs)
                y_preds.extend(preds)
                y_true.extend(y.cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        acc = accuracy_score(y_true, y_preds) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_preds, average="binary", zero_division=0)
        auc = roc_auc_score(y_true, y_probs) * 100
        eer = compute_eer(y_true, y_probs)

        print(f"Epoch {epoch:02d}/16 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.4f} | AUC: {auc:.2f}% | EER: {eer:.2f}%")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), str(best_weights_path))
            print(f"  --> Saved new best checkpoint to {best_weights_path.name} (F1: {f1:.4f})")

    print("=" * 65)
    print(f"TRAINING COMPLETE! Best Model Checkpoint Saved: {best_weights_path}")
    print("=" * 65)

if __name__ == "__main__":
    train_unified_model()
