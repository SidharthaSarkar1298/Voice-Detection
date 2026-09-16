/**
 * SIH26104 Voice Shield: Client-Side Audio Streamer
 * Captures microphone PCM audio and streams to WebSocket in real-time.
 */

class AudioStreamer {
  constructor(wsUrl, onTelemetryCallback) {
    this.wsUrl = wsUrl;
    this.onTelemetry = onTelemetryCallback;
    this.socket = null;
    this.audioContext = null;
    this.mediaStream = null;
    this.processor = null;
    this.isStreaming = false;
    this.targetSampleRate = 16000;
  }

  async start(claimedSpeakerId) {
    if (this.isStreaming) return;

    try {
      // 1. Initialize WebSocket
      this.socket = new WebSocket(this.wsUrl);
      this.socket.binaryType = "arraybuffer";

      this.socket.onopen = () => {
        console.log(" Telephony Stream WebSocket Connected");
        this.socket.send(JSON.stringify({
          type: "SET_SPEAKER",
          speaker_id: claimedSpeakerId
        }));
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (this.onTelemetry) {
            this.onTelemetry(data);
          }
        } catch (e) {
          console.error("Error parsing telemetry:", e);
        }
      };

      this.socket.onerror = (err) => {
        console.error("WebSocket Error:", err);
      };

      this.socket.onclose = () => {
        console.log("WebSocket Disconnected");
        this.stop();
      };

      // 2. Request Mic Stream
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: true
        }
      });

      // 3. Set up AudioContext
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtx({ sampleRate: this.targetSampleRate });
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);

      // 4. Create ScriptProcessorNode (bufferSize 4096 = ~256ms per packet at 16kHz)
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

      this.processor.onaudioprocess = (e) => {
        if (!this.isStreaming || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
          return;
        }
        const inputData = e.inputBuffer.getChannelData(0);
        let pcmToSend = inputData;
        const actualRate = this.audioContext.sampleRate;

        // Ensure clean 16kHz resampling if browser hardware audio runs at 44.1k / 48k
        if (actualRate && actualRate !== this.targetSampleRate) {
          const ratio = actualRate / this.targetSampleRate;
          const newLength = Math.round(inputData.length / ratio);
          const resampled = new Float32Array(newLength);
          for (let i = 0; i < newLength; i++) {
            const origIndex = i * ratio;
            const indexFloor = Math.floor(origIndex);
            const indexCeil = Math.min(inputData.length - 1, indexFloor + 1);
            const fraction = origIndex - indexFloor;
            resampled[i] = inputData[indexFloor] * (1 - fraction) + inputData[indexCeil] * fraction;
          }
          pcmToSend = resampled;
        }

        const pcmBuffer = new Float32Array(pcmToSend);
        this.socket.send(pcmBuffer.buffer);
      };

      source.connect(this.processor);
      const muteGain = this.audioContext.createGain();
      muteGain.gain.value = 0.0;
      this.processor.connect(muteGain);
      muteGain.connect(this.audioContext.destination);

      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      this.isStreaming = true;
      console.log(" Live microphone streaming armed and running.");
      return true;

    } catch (err) {
      console.error("Failed to start audio stream:", err);
      alert("Microphone Access Error: " + err.message);
      this.stop();
      return false;
    }
  }

  stop() {
    this.isStreaming = false;

    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close();
      this.audioContext = null;
    }

    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.close();
      this.socket = null;
    }

    console.log("[AudioStreamer] Audio stream stopped.");
  }

  updateClaimedSpeaker(speakerId) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: "SET_SPEAKER",
        speaker_id: speakerId
      }));
    }
  }
}

window.AudioStreamer = AudioStreamer;
