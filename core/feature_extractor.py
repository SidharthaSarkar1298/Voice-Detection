import numpy as np
import torch
import torchaudio
import librosa
from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, TARGET_FRAMES

class FeatureExtractor:
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, target_frames=TARGET_FRAMES):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=self.n_mels
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def preprocess_audio(self, wav_tensor, sr):
        """
        Ensures 1D mono, resampled to 16kHz, peak-normalized.
        """
        if isinstance(wav_tensor, np.ndarray):
            wav_tensor = torch.from_numpy(wav_tensor).float()
            
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.mean(dim=0)
            
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav_tensor = resampler(wav_tensor)
            
        # Peak normalization
        max_val = torch.max(torch.abs(wav_tensor))
        if max_val > 1e-6:
            wav_tensor = wav_tensor / max_val
            
        return wav_tensor

    def extract_mel_spectrogram(self, wav_tensor):
        """
        Computes Mel spectrogram formatted as (1, N_MELS, TARGET_FRAMES)
        """
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
            
        mel_spec = self.mel_transform(wav_tensor)
        mel_db = self.amp_to_db(mel_spec).squeeze(0).numpy()
        
        # Pad or crop to target_frames
        n_mels, t_frames = mel_db.shape
        if t_frames < self.target_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.target_frames - t_frames)), mode='constant', constant_values=-80.0)
        elif t_frames > self.target_frames:
            mel_db = mel_db[:, :self.target_frames]
            
        return mel_db.astype(np.float32)

    def extract_prosody_features(self, wav_np):
        """
        Extracts speech biomarkers: pitch (F0), jitter, shimmer, pause ratio, spectral flatness.
        Returns a dict and an anomaly score [0.0 - 1.0].
        """
        if len(wav_np) < 512:
            return {
                "f0_mean": 0.0, "f0_std": 0.0, "jitter": 0.0, "shimmer": 0.0,
                "silence_ratio": 1.0, "spectral_flatness": 0.0, "anomaly_score": 0.5
            }
            
        # 1. Pitch (F0) estimation using Yin algorithm
        try:
            f0 = librosa.yin(wav_np, fmin=50, fmax=500, sr=self.sample_rate)
            voiced = f0[f0 > 0]
            if len(voiced) > 5:
                f0_mean = float(np.mean(voiced))
                f0_std = float(np.std(voiced))
                f0_range = float(np.percentile(voiced, 95) - np.percentile(voiced, 5))
            else:
                f0_mean, f0_std, f0_range = 150.0, 0.0, 0.0
        except Exception:
            f0_mean, f0_std, f0_range = 150.0, 0.0, 0.0

        # 2. Local Jitter (Pitch perturbation)
        # Synthetic speech typically has unnaturally low jitter (too robotic/perfect) or extreme pitch jumps
        if len(voiced) > 10:
            diffs = np.abs(np.diff(voiced))
            jitter = float(np.mean(diffs) / (f0_mean + 1e-6))
        else:
            jitter = 0.0

        # 3. Local Shimmer (Amplitude perturbation)
        frame_len = 512
        hop = 256
        frames = librosa.util.frame(wav_np, frame_length=frame_len, hop_length=hop)
        amplitudes = np.max(np.abs(frames), axis=0)
        amp_voiced = amplitudes[amplitudes > 0.02]
        if len(amp_voiced) > 10:
            amp_diffs = np.abs(np.diff(amp_voiced))
            shimmer = float(np.mean(amp_diffs) / (np.mean(amp_voiced) + 1e-6))
        else:
            shimmer = 0.0

        # 4. Silence & Pause Cadence
        rms = librosa.feature.rms(y=wav_np, frame_length=frame_len, hop_length=hop)[0]
        silence_thresh = 0.02
        silence_ratio = float(np.mean(rms < silence_thresh))

        # 5. Spectral Flatness & Centroid
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=wav_np)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=wav_np, sr=self.sample_rate)))

        # Prosodic Anomaly Calculation
        # Neural TTS usually manifests:
        # - Excessively smooth pitch (f0_std < 12 Hz) or unnatural monotonic pitch
        # - Low micro-jitter (< 0.008) or erratic pitch jumps
        # - High artificial silence zeroing (silence_ratio > 0.50 with sudden cutoffs)
        # - Spectral flatness deviations (vocoder smoothing or high-freq sizzle)
        anomaly_points = 0.0
        if f0_std < 15.0 or f0_std > 80.0:
            anomaly_points += 0.30
        if jitter < 0.008 or jitter > 0.08:
            anomaly_points += 0.25
        if shimmer < 0.02 or shimmer > 0.35:
            anomaly_points += 0.20
        if silence_ratio > 0.55:
            anomaly_points += 0.15
        if flatness > 0.15:
            anomaly_points += 0.10

        anomaly_score = min(1.0, max(0.0, anomaly_points))

        return {
            "f0_mean_hz": round(f0_mean, 1),
            "f0_std_hz": round(f0_std, 1),
            "f0_range_hz": round(f0_range, 1),
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4),
            "silence_ratio": round(silence_ratio, 3),
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_flatness": round(flatness, 4),
            "anomaly_score": round(anomaly_score, 3)
        }

    def extract_speaker_embedding(self, wav_np):
        """
        Extracts a spectro-pitch biometric identity vector:
        combining 32-band Mel profile, pitch statistics, and spectral contrast.
        """
        if len(wav_np) < 512:
            return np.zeros(41, dtype=np.float32)
            
        mels = librosa.feature.melspectrogram(y=wav_np, sr=self.sample_rate, n_mels=32)
        mel_db = librosa.power_to_db(mels, ref=np.max)
        spec_profile = np.mean(mel_db, axis=1)
        
        try:
            f0 = librosa.yin(wav_np, fmin=60, fmax=400, sr=self.sample_rate)
            voiced = f0[f0 > 0]
            p_mean = float(np.mean(voiced)) if len(voiced) else 150.0
            p_std = float(np.std(voiced)) if len(voiced) else 20.0
        except Exception:
            p_mean, p_std = 150.0, 20.0
            
        contrast = np.mean(librosa.feature.spectral_contrast(y=wav_np, sr=self.sample_rate), axis=1)
        vec = np.concatenate([spec_profile, [p_mean / 10.0, p_std], contrast])
        
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
            
        return vec.astype(np.float32)
