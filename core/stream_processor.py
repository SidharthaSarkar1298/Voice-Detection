import numpy as np
import collections
from config import SAMPLE_RATE, CHUNK_DURATION

class AudioStreamBuffer:
    """
    Rolling ring buffer for near-real-time streaming audio ingestion.
    Preserves audio in memory only (Zero Disk Retention / DPDP Compliance).
    """
    def __init__(self, sample_rate=SAMPLE_RATE, window_duration=CHUNK_DURATION):
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.max_samples = int(sample_rate * window_duration)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.total_samples_received = 0

    def add_chunk(self, chunk_np: np.ndarray):
        """
        Appends new PCM audio samples to the ring buffer.
        """
        if chunk_np.dtype != np.float32:
            chunk_np = chunk_np.astype(np.float32)
            
        # If stereo, average to mono
        if chunk_np.ndim > 1:
            chunk_np = np.mean(chunk_np, axis=0)
            
        self.buffer.extend(chunk_np)
        self.total_samples_received += len(chunk_np)

    def is_ready(self, min_samples=None):
        """
        Returns True if buffer has accumulated enough audio for a meaningful inference pass.
        """
        req = min_samples or int(self.sample_rate * 0.75) # at least 0.75 sec
        return len(self.buffer) >= req

    def get_current_window(self):
        """
        Returns the most recent audio window as a 1D float32 numpy array.
        """
        arr = np.array(self.buffer, dtype=np.float32)
        if len(arr) < self.max_samples:
            # Pad with zeros if starting call
            arr = np.pad(arr, (self.max_samples - len(arr), 0), mode='constant')
        return arr

    def reset(self):
        self.buffer.clear()
        self.total_samples_received = 0
