import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from config import DATA_DIR, MODELS_DIR, DEVICE, TARGET_FRAMES
from core.feature_extractor import FeatureExtractor
from core.acoustic_model import VoiceShieldAcousticCNN

class VoiceShieldDataset(Dataset):
    def __init__(self, df, feature_extractor, augment=True):
        self.df = df.reset_index(drop=True)
        self.fe = feature_extractor
        self.augment = augment
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["path"]
        label = float(row["label"])

        if wav_path not in self.cache:
            wav, sr = sf.read(wav_path)
            wav_tensor = self.fe.preprocess_audio(wav, sr)
            mel = self.fe.extract_mel_spectrogram(wav_tensor)
            self.cache[wav_path] = mel
        else:
            mel = self.cache[wav_path].copy()

        # Augmentation
        if self.augment:
            # Time masking
            if np.random.rand() < 0.35:
                t = np.random.randint(0, mel.shape[1] - 8)
                mel[:, t:t+8] = -80.0
            # Frequency masking
            if np.random.rand() < 0.35:
                f = np.random.randint(0, mel.shape[0] - 8)
                mel[f:f+8, :] = -80.0
            # Gaussian noise
            if np.random.rand() < 0.30:
                mel = mel + np.random.normal(0, 0.5, mel.shape).astype(np.float32)

        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0) # (1, N_MELS, TARGET_FRAMES)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y

def train_model():
    print("=" * 60)
    print(f"SIH26104 Voice Shield: Training Acoustic CNN on {DEVICE.upper()}")
    print("=" * 60)

    manifest_path = DATA_DIR / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    df = pd.read_csv(manifest_path)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    fe = FeatureExtractor()
    dataset = VoiceShieldDataset(df, fe, augment=True)

    # Train / Val split (80% / 20%)
    n_total = len(dataset)
    n_train = int(0.80 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = VoiceShieldAcousticCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_f1 = 0.0
    best_model_path = MODELS_DIR / "voice_shield_cnn.pt"

    for epoch in range(1, 16):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            # Label smoothing
            y_b_smooth = y_b * 0.90 + 0.05

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b_smooth)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        all_preds, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(x_b)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y_b.cpu().numpy().flatten())

        val_acc = accuracy_score(all_targets, all_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
        try:
            val_auc = roc_auc_score(all_targets, all_probs)
        except Exception:
            val_auc = 0.5

        print(f"Epoch [{epoch:02d}/15] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.1f}% | "
              f"Precision: {val_prec:.3f} | Recall: {val_rec:.3f} | F1: {val_f1:.3f} | AUC: {val_auc:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> Saved new best model checkpoint to {best_model_path}")

    print("=" * 60)
    print(f"Training Complete! Best Validation F1: {best_val_f1:.4f}")
    print(f"Production Model Artifact: {best_model_path}")
    print("=" * 60)
    return best_model_path

if __name__ == "__main__":
    train_model()
