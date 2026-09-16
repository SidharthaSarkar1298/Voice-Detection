import os
import sys
import io
import zipfile
import asyncio
from pathlib import Path
import numpy as np
import requests
import soundfile as sf
import librosa
import edge_tts

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

TEST_DIR = root_dir / "test"
REAL_TEST_DIR = TEST_DIR / "real"
FAKE_TEST_DIR = TEST_DIR / "fake"
REAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
FAKE_TEST_DIR.mkdir(parents=True, exist_ok=True)

async def generate_fake_voices():
    print("Generating authentic neural deepfake and voice cloning samples from the web...")
    
    # Text prompts simulating realistic social engineering, banking, and executive impersonation attacks
    prompts = [
        # Indian English / Executive
        ("This is Rajesh Sharma from the executive board. Please process the urgent wire transfer immediately.", "en-IN-PrabhatNeural", "fake_01_indian_male_neural_ceo.wav"),
        ("Hello, this is Anita Desai. I am authorizing the release of funds for the vendor payment.", "en-IN-NeerjaNeural", "fake_02_indian_female_neural_cfo.wav"),
        ("I need you to bypass the standard secondary verification protocol for this confidential acquisition.", "en-IN-PrabhatNeural", "fake_03_indian_male_urgent_transfer.wav"),
        ("Please confirm if the twenty-five lakh rupees credit has been processed to our offshore account.", "en-IN-NeerjaNeural", "fake_04_indian_female_offshore_query.wav"),
        
        # US English / Executive & High-Risk
        ("Good morning, this is the chief financial officer calling. Please confirm receipt of the revised invoice.", "en-US-GuyNeural", "fake_05_us_male_neural_cfo.wav"),
        ("I am currently in an executive meeting and cannot access email. Please initiate the transfer right now.", "en-US-JennyNeural", "fake_06_us_female_neural_exec.wav"),
        ("This is Vikram Singh from corporate treasury. We need immediate clearance on the security deposit.", "en-US-ChristopherNeural", "fake_07_us_male_treasury_scam.wav"),
        ("The board has approved the emergency funds allocation. Forward the transaction confirmation receipt.", "en-US-AriaNeural", "fake_08_us_female_emergency_wire.wav"),
        
        # British English & Global Accents
        ("Hello, I am calling regarding the privileged access approval for the core banking database.", "en-GB-RyanNeural", "fake_09_gb_male_banking_access.wav"),
        ("Please verify the routing code and transfer twenty thousand pounds to the designated holding account.", "en-GB-SoniaNeural", "fake_10_gb_female_routing_wire.wav"),
        
        # Hindi & Multilingual Impersonation
        ("Namaste, main bank headquarters se bol raha hoon. Kripya is transaction ko turant approve karein.", "hi-IN-MadhurNeural", "fake_11_hindi_male_bank_approval.wav"),
        ("Hello, kya aapne accounts department ka naya authorization letter verify kar liya hai?", "hi-IN-SwaraNeural", "fake_12_hindi_female_accounts_verify.wav"),
        
        # Additional Diverse Neural Clones
        ("Security token authentication is currently offline. You are authorized to proceed with the wire transfer.", "en-US-EricNeural", "fake_13_us_male_token_bypass.wav"),
        ("Please ensure the confidential wire transfer is dispatched before the daily market closure.", "en-IN-NeerjaNeural", "fake_14_indian_female_market_wire.wav"),
        ("This is an automated priority voice notice from the corporate security operations center.", "en-US-GuyNeural", "fake_15_us_male_soc_alert.wav"),
    ]
    
    for text, voice, fname in prompts:
        out_path = FAKE_TEST_DIR / fname
        temp_mp3 = FAKE_TEST_DIR / f"temp_{fname}.mp3"
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(temp_mp3))
            
            # Load and convert to 16kHz Mono WAV
            y, sr = librosa.load(str(temp_mp3), sr=16000)
            
            # Save pristine neural synthetic WAV
            sf.write(str(out_path), y, 16000)
            if temp_mp3.exists():
                temp_mp3.unlink()
            print(f"  [+] Generated Fake: {fname} ({voice})")
        except Exception as e:
            print(f"  [-] Failed generating {fname}: {e}")

def fetch_real_voices():
    print("\nFetching authentic human speech dataset from the internet...")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
    
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"Failed to fetch speech commands: {r.status_code}")
        return

    print("Reading and assembling authentic multi-speaker human utterances...")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Filter speech commands files by distinct speaker IDs (filename format: speaker_nohash_X.wav)
    all_files = [f for f in z.namelist() if f.endswith(".wav") and not f.startswith("__")]
    
    # Group by speaker
    speakers = {}
    for f in all_files:
        parts = Path(f).stem.split("_nohash_")
        if len(parts) == 2:
            spk = parts[0]
            if spk not in speakers:
                speakers[spk] = []
            speakers[spk].append(f)
            
    print(f"Found {len(speakers)} distinct real human speakers in corpus.")
    
    # Create 15 multi-syllabic human voice samples by chaining utterances per speaker (giving realistic 2.5 - 3.5s sentences)
    speaker_list = list(speakers.keys())[:15]
    
    for idx, spk in enumerate(speaker_list):
        files_for_spk = speakers[spk][:4] # take 3-4 words spoken by this human
        combined_audio = []
        
        for f in files_for_spk:
            with z.open(f) as audio_f:
                data, sr = sf.read(io.BytesIO(audio_f.read()))
                if sr != 16000:
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                combined_audio.append(data)
                # add short natural human inter-word pause (150ms)
                combined_audio.append(np.zeros(int(16000 * 0.15), dtype=np.float32))
                
        if combined_audio:
            y = np.concatenate(combined_audio).astype(np.float32)
            # Normalize peak
            max_v = np.max(np.abs(y))
            if max_v > 0:
                y = y / max_v * 0.92
                
            out_name = f"real_{idx+1:02d}_human_speaker_{spk}.wav"
            out_file = REAL_TEST_DIR / out_name
            sf.write(str(out_file), y, 16000)
            print(f"  [+] Extracted Real: {out_name} (Speaker {spk}, Duration: {len(y)/16000:.2f}s)")

if __name__ == "__main__":
    print("=" * 60)
    print("SIH26104 Voice Shield: Building ASVspoof Benchmark Test Sets")
    print("=" * 60)
    asyncio.run(generate_fake_voices())
    fetch_real_voices()
    
    n_real = len(list(REAL_TEST_DIR.glob("*.wav")))
    n_fake = len(list(FAKE_TEST_DIR.glob("*.wav")))
    print("=" * 60)
    print(f"Test Dataset Assembled:")
    print(f" - Real (Bonafide) Voices in 'test/real/': {n_real} files")
    print(f" - Fake (Spoofed) Voices in 'test/fake/':   {n_fake} files")
    print("=" * 60)
