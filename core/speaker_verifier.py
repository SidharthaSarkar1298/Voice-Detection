import os
import json
import numpy as np
from pathlib import Path
from config import PROFILES_DIR

class SpeakerVerifier:
    """
    Biometric Speaker Verification Engine for detecting Identity Impersonation.
    Stores and compares acoustic identity vectors against enrolled profiles.
    """
    def __init__(self, profiles_dir=PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profiles = {}
        self.load_all_profiles()

    def enroll_speaker(self, speaker_id: str, full_name: str, designation: str, embedding: np.ndarray, metadata: dict = None):
        """
        Enrolls a verified voice profile.
        """
        # Ensure L2 normalized
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        profile_data = {
            "speaker_id": speaker_id,
            "full_name": full_name,
            "designation": designation,
            "embedding": embedding.tolist(),
            "metadata": metadata or {}
        }
        
        file_path = self.profiles_dir / f"{speaker_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)
            
        self.profiles[speaker_id] = profile_data
        return profile_data

    def load_all_profiles(self):
        """
        Loads all stored speaker profiles from disk.
        """
        self.profiles = {}
        for p_file in self.profiles_dir.glob("*.json"):
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.profiles[data["speaker_id"]] = data
            except Exception as e:
                print(f"Failed to load profile {p_file}: {e}")
        return self.profiles

    def verify_speaker(self, claimed_speaker_id: str, test_embedding: np.ndarray, threshold: float = 0.965, k: float = 45.0):
        """
        Calculates cosine similarity and calibrated biometric match probability.
        Returns:
            similarity (float): [0.0 - 1.0]
            mismatch_score (float): [0.0 - 1.0] (higher = higher risk of impersonation)
            is_match (bool)
        """
        if not claimed_speaker_id or claimed_speaker_id not in self.profiles:
            return {
                "similarity": 0.5,
                "mismatch_score": 0.3,
                "is_match": True,
                "note": "Unenrolled / general caller"
            }

        enrolled_emb = np.array(self.profiles[claimed_speaker_id]["embedding"], dtype=np.float32)
        
        # Raw Cosine similarity
        norm_test = np.linalg.norm(test_embedding)
        norm_enrolled = np.linalg.norm(enrolled_emb)
        
        if norm_test < 1e-6 or norm_enrolled < 1e-6:
            raw_sim = 0.0
        else:
            raw_sim = float(np.dot(test_embedding, enrolled_emb) / (norm_test * norm_enrolled))

        # Calibrated Sigmoid Match Probability
        calibrated_match = float(1.0 / (1.0 + np.exp(-k * (raw_sim - threshold))))
        calibrated_match = max(0.0, min(1.0, calibrated_match))
        
        mismatch_score = 1.0 - calibrated_match
        is_match = calibrated_match >= 0.55

        return {
            "raw_similarity": round(raw_sim, 4),
            "similarity": round(calibrated_match, 3),
            "mismatch_score": round(mismatch_score, 3),
            "is_match": is_match,
            "claimed_name": self.profiles[claimed_speaker_id]["full_name"],
            "designation": self.profiles[claimed_speaker_id]["designation"]
        }
