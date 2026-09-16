import os
import sys
import json
import io
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import SAMPLE_RATE, DEVICE, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR
from core.feature_extractor import FeatureExtractor
from core.acoustic_model import VoiceShieldAcousticCNN
from core.stream_processor import AudioStreamBuffer

app = FastAPI(
    title="VOICE SHIELD: ASVspoof Real-Time Voice Authenticity & Deepfake Detector",
    version="2.0.0"
)

# Static and test directories
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

test_dir = root_dir / "test"
if test_dir.exists():
    app.mount("/test", StaticFiles(directory=str(test_dir)), name="test")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Core Feature Extractor and Model
feature_extractor = FeatureExtractor()
acoustic_model = VoiceShieldAcousticCNN().to(DEVICE)
model_path = MODELS_DIR / "voice_shield_cnn.pt"

if model_path.exists():
    acoustic_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    acoustic_model.eval()
    print(f" Loaded Voice Shield ASVspoof Acoustic CNN from {model_path} on {DEVICE.upper()}")
else:
    print(f"  Warning: Model weights not found at {model_path}.")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    real_files = [f.name for f in sorted((test_dir / "real").glob("*.wav"))] if (test_dir / "real").exists() else []
    fake_files = [f.name for f in sorted((test_dir / "fake").glob("*.wav"))] if (test_dir / "fake").exists() else []
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "real_files": real_files,
            "fake_files": fake_files
        }
    )

@app.get("/api/v1/test-list")
async def get_test_list():
    real_files = [f.name for f in sorted((test_dir / "real").glob("*.wav"))] if (test_dir / "real").exists() else []
    fake_files = [f.name for f in sorted((test_dir / "fake").glob("*.wav"))] if (test_dir / "fake").exists() else []
    return JSONResponse(content={
        "real": real_files,
        "fake": fake_files
    })

@app.post("/api/v1/analyze")
async def analyze_voice(
    file: Optional[UploadFile] = File(None),
    sample_type: Optional[str] = Form(None), # 'real' or 'fake'
    filename: Optional[str] = Form(None)
):
    try:
        audio_url = None
        # Ingest from test folder or uploaded file
        if file is not None and file.filename:
            content = await file.read()
            wav, sr = sf.read(io.BytesIO(content))
            display_name = file.filename
        elif sample_type and filename:
            file_path = test_dir / sample_type / filename
            if not file_path.exists():
                return JSONResponse(status_code=404, content={"error": f"Sample not found: {filename}"})
            wav, sr = sf.read(str(file_path))
            display_name = filename
            audio_url = f"/test/{sample_type}/{filename}"
        else:
            # Default to first real file
            first_real = next((test_dir / "real").glob("*.wav"), None)
            if first_real:
                wav, sr = sf.read(str(first_real))
                display_name = first_real.name
                audio_url = f"/test/real/{first_real.name}"
            else:
                return JSONResponse(status_code=400, content={"error": "No audio input provided"})

        # Preprocess
        wav_tensor = feature_extractor.preprocess_audio(wav, sr)
        wav_np = wav_tensor.numpy()
        duration_sec = len(wav_np) / SAMPLE_RATE

        # 1. 80-Band Mel-Spectrogram & PyTorch Model Inference
        mel = feature_extractor.extract_mel_spectrogram(wav_tensor)
        mel_in = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = acoustic_model(mel_in)
            spoof_prob = float(torch.sigmoid(logits).item())

        # 2. Prosodic & Speech Biomarker Extraction
        prosody = feature_extractor.extract_prosody_features(wav_np)

        # Decision Logic
        is_spoof = spoof_prob >= 0.50
        verdict = "SYNTHETIC SPOOF" if is_spoof else "AUTHENTIC SPEECH"
        confidence_percent = round((spoof_prob if is_spoof else (1.0 - spoof_prob)) * 100, 1)

        if is_spoof:
            badge_class = "danger"
            summary_title = "Synthetic / AI-Cloned Audio Detected"
            summary_desc = f"Neural vocoder artifacts and synthetic spectral discontinuities detected with {confidence_percent}% confidence."
        else:
            badge_class = "safe"
            summary_title = "Authentic Human Speech Verified"
            summary_desc = f"Organic vocal tract harmonics and natural prosodic cadence verified with {confidence_percent}% confidence."

        # 3. Time-Series Sliding Window Timeline
        window_size = int(SAMPLE_RATE * 1.0)
        step_size = int(SAMPLE_RATE * 0.4)
        timeline = []

        for start_idx in range(0, max(1, len(wav_np) - window_size + 1), step_size):
            chunk = wav_np[start_idx:start_idx + window_size]
            t_sec = round(start_idx / SAMPLE_RATE, 2)
            chunk_mel = feature_extractor.extract_mel_spectrogram(torch.from_numpy(chunk))
            chunk_in = torch.tensor(chunk_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                c_prob = float(torch.sigmoid(acoustic_model(chunk_in)).item())
            timeline.append({
                "time_sec": t_sec,
                "spoof_prob": round(c_prob * 100, 1)
            })

        # Downsample waveform for Plotly
        max_points = 900
        step = max(1, len(wav_np) // max_points)
        downsampled_waveform = wav_np[::step].tolist()

        return JSONResponse(content={
            "success": True,
            "filename": display_name,
            "audio_url": audio_url,
            "duration_sec": round(duration_sec, 2),
            "verdict": verdict,
            "is_spoof": is_spoof,
            "spoof_prob": round(spoof_prob * 100, 1),
            "confidence_percent": confidence_percent,
            "badge_class": badge_class,
            "summary_title": summary_title,
            "summary_desc": summary_desc,
            "prosody": prosody,
            "waveform": downsampled_waveform,
            "mel_spectrogram": mel.tolist(),
            "timeline": timeline
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/v1/benchmark")
async def run_benchmark():
    """
    Evaluates all files in test/real and test/fake, computing ASVspoof metrics:
    Accuracy, Precision, Recall, F1, and Equal Error Rate (EER).
    """
    try:
        real_files = sorted((test_dir / "real").glob("*.wav"))
        fake_files = sorted((test_dir / "fake").glob("*.wav"))
        eleven_files = sorted((test_dir / "Fake Voices 11 Labs").glob("*.wav"))

        results = []
        y_true, y_pred, y_prob = [], [], []

        # Test Real (label 0)
        for rf in real_files:
            wav, sr = sf.read(str(rf))
            wav_tensor = feature_extractor.preprocess_audio(wav, sr)
            mel = feature_extractor.extract_mel_spectrogram(wav_tensor)
            mel_in = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = float(torch.sigmoid(acoustic_model(mel_in)).item())
            
            pred = 1 if prob >= 0.5 else 0
            y_true.append(0)
            y_pred.append(pred)
            y_prob.append(prob)
            gt_name = "BONAFIDE (IndicVoices)" if "indic" in rf.name.lower() else "BONAFIDE (Human)"
            results.append({
                "file": rf.name,
                "ground_truth": gt_name,
                "spoof_prob": round(prob * 100, 1),
                "verdict": "SPOOF" if pred == 1 else "BONAFIDE",
                "correct": pred == 0
            })

        # Test General Fake (label 1)
        for ff in fake_files:
            wav, sr = sf.read(str(ff))
            wav_tensor = feature_extractor.preprocess_audio(wav, sr)
            mel = feature_extractor.extract_mel_spectrogram(wav_tensor)
            mel_in = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = float(torch.sigmoid(acoustic_model(mel_in)).item())
            
            pred = 1 if prob >= 0.5 else 0
            y_true.append(1)
            y_pred.append(pred)
            y_prob.append(prob)
            gt_fake = "SPOOF (WaveFake)" if "wavefake" in ff.name.lower() else "SPOOF (EdgeTTS)"
            results.append({
                "file": ff.name,
                "ground_truth": gt_fake,
                "spoof_prob": round(prob * 100, 1),
                "verdict": "SPOOF" if pred == 1 else "BONAFIDE",
                "correct": pred == 1
            })

        # Test ElevenLabs Fake (label 1)
        for ef in eleven_files:
            wav, sr = sf.read(str(ef))
            wav_tensor = feature_extractor.preprocess_audio(wav, sr)
            mel = feature_extractor.extract_mel_spectrogram(wav_tensor)
            mel_in = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = float(torch.sigmoid(acoustic_model(mel_in)).item())
            
            pred = 1 if prob >= 0.5 else 0
            y_true.append(1)
            y_pred.append(pred)
            y_prob.append(prob)
            results.append({
                "file": ef.name,
                "ground_truth": "SPOOF (11Labs)",
                "spoof_prob": round(prob * 100, 1),
                "verdict": "SPOOF" if pred == 1 else "BONAFIDE",
                "correct": pred == 1
            })

        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = round(correct / total * 100, 2)

        return JSONResponse(content={
            "success": True,
            "total_tested": total,
            "correct": correct,
            "accuracy": accuracy,
            "real_count": len(real_files),
            "fake_count": len(fake_files) + len(eleven_files),
            "results": results
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/api/v1/stream/ws")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    stream_buffer = AudioStreamBuffer()

    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                chunk = np.frombuffer(data["bytes"], dtype=np.float32)
                stream_buffer.add_chunk(chunk)

                if stream_buffer.is_ready():
                    window = stream_buffer.get_current_window()
                    wav_tensor = feature_extractor.preprocess_audio(window, sr=16000)
                    mel = feature_extractor.extract_mel_spectrogram(wav_tensor)
                    mel_in = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        spoof_prob = float(torch.sigmoid(acoustic_model(mel_in)).item())

                    prosody = feature_extractor.extract_prosody_features(window)
                    is_spoof = spoof_prob >= 0.50
                    conf = spoof_prob if is_spoof else (1.0 - spoof_prob)
                    confidence_percent = round(conf * 100, 1)

                    # Downsample waveform for live Plotly visualization (approx 250 points)
                    step = max(1, len(window) // 250)
                    waveform_samples = [round(float(v), 4) for v in window[::step]]

                    # Subsample mel frames along time axis (80 mel bands x ~100 frames)
                    mel_sampled = [[round(float(val), 2) for val in row[::2]] for row in mel]

                    # Measure RMS audio energy to distinguish silence vs vocal speech
                    rms_energy = float(np.sqrt(np.mean(window ** 2)))
                    is_speaking = bool(rms_energy > 0.006)

                    await websocket.send_json({
                        "type": "STREAM_UPDATE",
                        "spoof_prob": round(spoof_prob * 100, 1),
                        "confidence_percent": confidence_percent,
                        "is_spoof": is_spoof,
                        "verdict": "SPOOF (FAKE VOICE)" if is_spoof else "BONAFIDE (REAL VOICE)",
                        "prosody": prosody,
                        "waveform": waveform_samples,
                        "mel_spectrogram": mel_sampled,
                        "rms_energy": round(rms_energy, 4),
                        "is_speaking": is_speaking
                    })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
