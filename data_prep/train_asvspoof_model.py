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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from config import MODELS_DIR, DEVICE, TARGET_FRAMES
from core.feature_extractor import FeatureExtractor
from core.acoustic_model import VoiceShieldAcousticCNN

class ASVSpoofDataset(Dataset):
    def __init__(self, file_list, labels, feature_extractor, augment=True):
        self.files = file_list
        self.labels = labels
        self.fe = feature_extractor
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        wav, sr = sf.read(str(path))
        wav_tensor = self.fe.preprocess_audio(wav, sr)
        mel = self.fe.extract_mel_spectrogram(wav_tensor)

        if self.augment:
            # Time masking
            if np.random.rand() < 0.35:
                t = np.random.randint(0, mel.shape[1] - 8)
                mel[:, t:t+8] = -80.0
            # Frequency masking
            if np.random.rand() < 0.35:
                f = np.random.randint(0, mel.shape[0] - 8)
                mel[f:f+8, :] = -80.0
            # Small Gaussian noise
            if np.random.rand() < 0.25:
                mel = mel + np.random.normal(0, 0.4, mel.shape).astype(np.float32)

        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y

def train_and_evaluate():
    print("=" * 60)
    print(f"Training ASVspoof Voice Shield CNN on {DEVICE.upper()}")
    print("=" * 60)

    fe = FeatureExtractor()

    train_real = list(Path("data/train_real").glob("*.wav"))
    train_fake = list(Path("data/train_fake").glob("*.wav"))
    test_real = list(Path("test/real").glob("*.wav"))
    test_fake = list(Path("test/fake").glob("*.wav"))

    train_files = train_real + train_fake
    train_labels = [0.0] * len(train_real) + [1.0] * len(train_fake)

    test_files = test_real + test_fake
    test_labels = [0.0] * len(test_real) + [1.0] * len(test_fake)

    print(f"Train set: {len(train_real)} Real, {len(train_fake)} Fake (Total: {len(train_files)})")
    print(f"Test set:  {len(test_real)} Real, {len(test_fake)} Fake (Total: {len(test_files)})")

    train_ds = ASVSpoofDataset(train_files, train_labels, fe, augment=True)
    test_ds = ASVSpoofDataset(test_files, test_labels, fe, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = VoiceShieldAcousticCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_test_acc = 0.0
    best_model_path = MODELS_DIR / "voice_shield_cnn.pt"

    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            y_smooth = y_b * 0.90 + 0.05
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_smooth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Evaluate on Test Set
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(DEVICE)
                logits = model(x_b)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                y_prob.extend(probs)
                y_pred.extend(preds)
                y_true.extend(y_b.numpy().flatten())

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        auc = roc_auc_score(y_true, y_prob)

        print(f"Epoch [{epoch:02d}/20] | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Test Acc: {acc*100:5.1f}% | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

        if acc >= best_test_acc and f1 >= 0.90:
            best_test_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> Saved checkpoint to {best_model_path}")

    print("=" * 60)
    print(f"ASVspoof Model Training Complete! Best Test Accuracy: {best_test_acc*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate()
