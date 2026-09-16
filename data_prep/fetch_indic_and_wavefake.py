import os
import sys
import io
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import fsspec
import pyarrow.parquet as pq

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

DATA_DIR = root_dir / "data"
TEST_DIR = root_dir / "test"

INDIC_REAL_DIR = DATA_DIR / "indic_real"
WAVEFAKE_DIR = DATA_DIR / "wavefake_fake"
TEST_REAL_DIR = TEST_DIR / "real"
TEST_FAKE_DIR = TEST_DIR / "fake"

for d in [INDIC_REAL_DIR, WAVEFAKE_DIR, TEST_REAL_DIR, TEST_FAKE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000

# Mapping of WaveFake codes to vocoder names
WF_VOCODER_MAP = {
    "WF1": "MelGAN",
    "WF2": "Parallel_WaveGAN",
    "WF3": "MultiBand_MelGAN",
    "WF4": "FullBand_MelGAN",
    "WF5": "HiFi_GAN",
    "WF6": "WaveGlow",
    "WF7": "Neural_Vocoder"
}

def fetch_indicvoices_real(target_count=160):
    print("=" * 60)
    print("FETCHING INDICVOICES (REAL INDIAN HUMAN SPEECH)")
    print("=" * 60)
    
    url = "https://huggingface.co/datasets/Bhasaanuvaad/IndicVoices_ST/resolve/main/Indic-En/hin/train-00000-of-00001.parquet"
    print(f"Streaming IndicVoices Parquet from: {url}")
    
    extracted = 0
    test_benchmarks = 0
    start_time = time.time()
    
    try:
        with fsspec.open(url, mode="rb") as f:
            pf = pq.ParquetFile(f)
            num_groups = pf.num_row_groups
            print(f"Remote IndicVoices Parquet has {num_groups} row groups ({pf.metadata.num_rows} total rows).")
            
            for rg_idx in range(num_groups):
                if extracted >= target_count:
                    break
                    
                rg = pf.read_row_group(rg_idx, columns=["chunked_audio_filepath", "duration"])
                data_dict = rg.to_pydict()
                audios = data_dict["chunked_audio_filepath"]
                durations = data_dict["duration"]
                
                for i, item in enumerate(audios):
                    if extracted >= target_count:
                        break
                    
                    audio_bytes = item.get("bytes")
                    dur = durations[i] if i < len(durations) else 3.0
                    
                    # Target clips between 2.0s and 6.0s
                    if not audio_bytes or dur < 1.8 or dur > 8.0:
                        continue
                        
                    try:
                        y, sr = sf.read(io.BytesIO(audio_bytes))
                        if sr != TARGET_SR:
                            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                        if y.ndim > 1:
                            y = np.mean(y, axis=1)
                            
                        # Peak normalize
                        peak = np.max(np.abs(y))
                        if peak > 1e-4:
                            y = (y / peak * 0.92).astype(np.float32)
                        else:
                            continue
                            
                        # Trim silence
                        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                        if len(y_trimmed) < int(TARGET_SR * 1.5):
                            continue
                            
                        extracted += 1
                        out_filename = f"indic_real_{extracted:04d}_hindi.wav"
                        out_path = INDIC_REAL_DIR / out_filename
                        sf.write(str(out_path), y_trimmed, TARGET_SR)
                        
                        # Save the first 10 files into test/real/ as official benchmark samples
                        if test_benchmarks < 10:
                            test_benchmarks += 1
                            bench_path = TEST_REAL_DIR / f"real_indic_{test_benchmarks:02d}_hindi.wav"
                            sf.write(str(bench_path), y_trimmed, TARGET_SR)
                            print(f"  [+] Indic Benchmark Saved: {bench_path.name} ({len(y_trimmed)/TARGET_SR:.2f}s)")
                            
                        if extracted % 25 == 0 or extracted == target_count:
                            print(f"  --> Extracted {extracted}/{target_count} IndicVoices Real samples... ({time.time() - start_time:.1f}s)")
                            
                    except Exception as e:
                        continue
                        
        print(f"Successfully extracted {extracted} IndicVoices real speech samples ({test_benchmarks} in test/real)!")
    except Exception as e:
        print(f"Error streaming IndicVoices: {e}")

def fetch_wavefake_synthetic(target_count=160):
    print("\n" + "=" * 60)
    print("FETCHING WAVEFAKE (ARTIFICIAL SYNTHETIC NEURAL VOICES)")
    print("=" * 60)
    
    url = "https://huggingface.co/datasets/ajaykarthick/wavefake-audio/resolve/main/data/partition0-00000-of-00001.parquet"
    print(f"Streaming WaveFake Parquet from: {url}")
    
    extracted = 0
    test_benchmarks = 0
    vocoder_counts = {}
    start_time = time.time()
    
    try:
        with fsspec.open(url, mode="rb") as f:
            pf = pq.ParquetFile(f)
            num_groups = pf.num_row_groups
            print(f"Remote WaveFake Parquet has {num_groups} row groups ({pf.metadata.num_rows} total rows).")
            
            for rg_idx in range(num_groups):
                if extracted >= target_count:
                    break
                    
                rg = pf.read_row_group(rg_idx)
                pyd = rg.to_pydict()
                audios = pyd["audio"]
                labels = pyd["real_or_fake"]
                
                for i, item in enumerate(audios):
                    if extracted >= target_count:
                        break
                        
                    label = labels[i]
                    # Only take synthetic vocoder clips (WF1 to WF7)
                    if label == "R" or not label.startswith("WF"):
                        continue
                        
                    audio_bytes = item.get("bytes")
                    if not audio_bytes:
                        continue
                        
                    try:
                        y, sr = sf.read(io.BytesIO(audio_bytes))
                        if sr != TARGET_SR:
                            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                        if y.ndim > 1:
                            y = np.mean(y, axis=1)
                            
                        # Normalize
                        peak = np.max(np.abs(y))
                        if peak > 1e-4:
                            y = (y / peak * 0.92).astype(np.float32)
                        else:
                            continue
                            
                        # Trim silence
                        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                        if len(y_trimmed) < int(TARGET_SR * 1.5):
                            continue
                            
                        # Take standard 2.5s - 4.5s clip
                        max_len = int(TARGET_SR * 4.5)
                        if len(y_trimmed) > max_len:
                            y_trimmed = y_trimmed[:max_len]
                            
                        vocoder_name = WF_VOCODER_MAP.get(label, "Neural_Vocoder")
                        vocoder_counts[vocoder_name] = vocoder_counts.get(vocoder_name, 0) + 1
                        
                        extracted += 1
                        out_filename = f"wavefake_{extracted:04d}_{vocoder_name}.wav"
                        out_path = WAVEFAKE_DIR / out_filename
                        sf.write(str(out_path), y_trimmed, TARGET_SR)
                        
                        # Save 10 diverse vocoder samples into test/fake/ as official benchmark samples
                        if test_benchmarks < 10:
                            test_benchmarks += 1
                            bench_path = TEST_FAKE_DIR / f"fake_wavefake_{test_benchmarks:02d}_{vocoder_name}.wav"
                            sf.write(str(bench_path), y_trimmed, TARGET_SR)
                            print(f"  [+] WaveFake Benchmark Saved: {bench_path.name} ({vocoder_name}, {len(y_trimmed)/TARGET_SR:.2f}s)")
                            
                        if extracted % 25 == 0 or extracted == target_count:
                            print(f"  --> Extracted {extracted}/{target_count} WaveFake Synthetic samples... ({time.time() - start_time:.1f}s)")
                            
                    except Exception as e:
                        continue
                        
        print(f"Successfully extracted {extracted} WaveFake synthetic samples ({test_benchmarks} in test/fake)!")
        print("Vocoder distribution:", vocoder_counts)
    except Exception as e:
        print(f"Error streaming WaveFake: {e}")

if __name__ == "__main__":
    fetch_indicvoices_real(target_count=160)
    fetch_wavefake_synthetic(target_count=160)
    print("\n" + "=" * 60)
    print("DATASET EXTRACTION COMPLETE!")
    print(f"IndicVoices Real:     {len(list(INDIC_REAL_DIR.glob('*.wav')))} training files")
    print(f"WaveFake Synthetic:   {len(list(WAVEFAKE_DIR.glob('*.wav')))} training files")
    print(f"Test Benchmarks Real: {len(list(TEST_REAL_DIR.glob('*.wav')))} total files")
    print(f"Test Benchmarks Fake: {len(list(TEST_FAKE_DIR.glob('*.wav')))} total files")
    print("=" * 60)
