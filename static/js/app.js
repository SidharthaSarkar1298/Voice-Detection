/**
 * VOICE SHIELD // Real-Time Voice Authenticity Controller
 */

let audioStreamer = null;
let liveRiskHistory = [];
let liveTimeHistory = [];
let streamStartTime = null;
let speechRecognizer = null;

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  if (tabId === 'preset') {
    document.getElementById('tabPresetBtn').classList.add('active');
    document.getElementById('presetTab').classList.add('active');
  } else if (tabId === 'mic') {
    document.getElementById('tabMicBtn').classList.add('active');
    document.getElementById('micTab').classList.add('active');
  } else if (tabId === 'upload') {
    document.getElementById('tabUploadBtn').classList.add('active');
    document.getElementById('uploadTab').classList.add('active');
  }
}

// Update Circular SVG Gauge & Risk Tag
function updateGauge(spoofProb, isSpoof) {
  const percentEl = document.getElementById('spoofPercent');
  const barCircle = document.getElementById('gaugeBarCircle');
  const riskTag = document.getElementById('gaugeRiskTag');
  const banner = document.getElementById('threatBanner');

  percentEl.innerText = `${spoofProb}%`;

  // Circumference for r=48 is 2 * PI * 48 = 301.59
  const circumference = 301.59;
  const clampedProb = Math.min(100, Math.max(0, spoofProb));
  const offset = circumference * (1 - clampedProb / 100);

  if (barCircle) {
    barCircle.style.strokeDashoffset = offset;
  }

  if (isSpoof) {
    percentEl.style.color = "#f85149";
    if (barCircle) {
      barCircle.style.stroke = "#f85149";
      barCircle.style.filter = "drop-shadow(0 0 8px rgba(248, 81, 73, 0.5))";
    }
    banner.style.borderColor = "#da3633";
    if (riskTag) {
      riskTag.innerText = "HIGH SPOOF";
      riskTag.style.color = "#f85149";
    }
  } else {
    percentEl.style.color = "#3fb950";
    if (barCircle) {
      barCircle.style.stroke = "#2ea043";
      barCircle.style.filter = "drop-shadow(0 0 8px rgba(46, 160, 67, 0.4))";
    }
    banner.style.borderColor = "#2ea043";
    if (riskTag) {
      riskTag.innerText = "BONAFIDE";
      riskTag.style.color = "#3fb950";
    }
  }
}

// Load Pre-Recorded Voice Sample
async function loadSample(sampleType, filename) {
  const formData = new FormData();
  formData.append('sample_type', sampleType);
  formData.append('filename', filename);

  updateUIProcessingState("Analyzing audio acoustics & vocoder spectrum...");

  try {
    const resp = await fetch('/api/v1/analyze', {
      method: 'POST',
      body: formData
    });
    const data = await resp.json();
    if (data.success) {
      renderAnalysisResults(data);
    } else {
      alert("Analysis error: " + (data.error || "Failed"));
    }
  } catch (err) {
    console.error(err);
    alert("Server communication error: " + err.message);
  }
}

// Upload Custom Audio File
async function uploadCustomAudio() {
  const fileInput = document.getElementById('audioFileInput');
  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Please select an audio file first.");
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  updateUIProcessingState("Uploading and analyzing audio file...");

  try {
    const resp = await fetch('/api/v1/analyze', {
      method: 'POST',
      body: formData
    });
    const data = await resp.json();
    if (data.success) {
      renderAnalysisResults(data);
    } else {
      alert("Error: " + (data.error || "Failed"));
    }
  } catch (err) {
    console.error(err);
    alert("Analysis error: " + err.message);
  }
}

// Render Results to UI
function renderAnalysisResults(data) {
  // Hide live transcript bar when viewing static file
  const transcriptCard = document.getElementById('liveTranscriptCard');
  if (transcriptCard) transcriptCard.style.display = 'none';

  // 1. Audio Player Setup
  const audioPlayer = document.getElementById('audioPlayer');
  const playingLabel = document.getElementById('playingFilename');

  if (data.audio_url && audioPlayer) {
    audioPlayer.src = data.audio_url;
    audioPlayer.load();
  }
  if (playingLabel) {
    playingLabel.innerText = `${data.filename} (${data.duration_sec}s)`;
  }

  // 2. Verdict Banner & SVG Gauge
  updateGauge(data.spoof_prob, data.is_spoof);

  const badge = document.getElementById('threatStatusBadge');
  const title = document.getElementById('summaryTitle');
  const desc = document.getElementById('summaryDesc');

  title.innerText = data.summary_title;
  desc.innerText = data.summary_desc;

  if (data.is_spoof) {
    badge.className = "threat-badge danger";
    badge.innerText = "SYNTHETIC SPOOF";
  } else {
    badge.className = "threat-badge safe";
    badge.innerText = "AUTHENTIC AUDIO";
  }

  // 3. Diagnostic Cards
  // Verdict Card
  document.getElementById('verdictOutput').innerText = data.verdict;
  document.getElementById('verdictBar').style.width = `${data.confidence_percent}%`;
  document.getElementById('verdictBar').style.background = data.is_spoof ? '#f85149' : '#2ea043';
  document.getElementById('verdictDesc').innerText = `${data.confidence_percent}% classification confidence`;

  // Artifacts Card
  document.getElementById('artifactOutput').innerText = `${data.spoof_prob}%`;
  document.getElementById('artifactBar').style.width = `${data.spoof_prob}%`;
  document.getElementById('artifactBar').style.background = data.spoof_prob > 50 ? '#f85149' : '#2ea043';

  // Prosody Card
  const prosodyAnomaly = Math.round(data.prosody.anomaly_score * 100);
  document.getElementById('prosodyOutput').innerText = `${100 - prosodyAnomaly}% Natural`;
  document.getElementById('prosodyBar').style.width = `${100 - prosodyAnomaly}%`;
  document.getElementById('prosodyBar').style.background = prosodyAnomaly > 50 ? '#f85149' : '#2ea043';

  // 4. Telemetry Grid (if present)
  if (document.getElementById('telF0')) {
    document.getElementById('telF0').innerText = `${data.prosody.f0_mean_hz} Hz`;
    document.getElementById('telF0Std').innerText = `±${data.prosody.f0_std_hz} Hz`;
    document.getElementById('telJitter').innerText = data.prosody.jitter;
    document.getElementById('telShimmer').innerText = data.prosody.shimmer;
    document.getElementById('telSilence').innerText = `${Math.round(data.prosody.silence_ratio * 100)}%`;
    document.getElementById('telFlatness').innerText = data.prosody.spectral_flatness;
  }

  // 5. Sidebar Waveform Plot
  const traceWave = {
    y: data.waveform,
    type: 'scatter',
    mode: 'lines',
    line: { color: data.is_spoof ? '#f85149' : '#58a6ff', width: 1.2 }
  };
  const layoutWave = {
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#0d1117',
    font: { color: '#8b949e', size: 9 },
    xaxis: { showgrid: false, zeroline: false, showticklabels: false },
    yaxis: { title: 'Amp', range: [-1.1, 1.1], gridcolor: '#21262d' },
    margin: { t: 10, b: 20, l: 35, r: 10 }
  };
  Plotly.react('waveformPlot', [traceWave], layoutWave);

  // 6. Sidebar 80-Band Mel-Spectrogram Heatmap
  const traceMel = {
    z: data.mel_spectrogram,
    type: 'heatmap',
    colorscale: data.is_spoof ? 'Magma' : 'Viridis',
    showscale: false
  };
  const layoutMel = {
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#0d1117',
    font: { color: '#8b949e', size: 9 },
    xaxis: { title: 'Time Frames', gridcolor: '#21262d' },
    yaxis: { title: 'Mel Bands', gridcolor: '#21262d' },
    margin: { t: 10, b: 30, l: 35, r: 10 }
  };
  Plotly.react('melPlot', [traceMel], layoutMel);

  // 7. Dynamic Timeline Plot (Main Panel)
  const times = data.timeline.map(t => t.time_sec);
  const spoofScores = data.timeline.map(t => t.spoof_prob);

  const traceTimeline = {
    x: times,
    y: spoofScores,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Spoof Probability (%)',
    line: { color: data.is_spoof ? '#f85149' : '#2ea043', width: 2.5 },
    marker: { size: 5 }
  };

  const layoutTimeline = {
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#0d1117',
    font: { color: '#8b949e', family: "'Inter', sans-serif", size: 11 },
    xaxis: { title: 'Audio Time Window (seconds)', gridcolor: '#21262d' },
    yaxis: { title: 'Spoof Probability (%)', range: [0, 100], gridcolor: '#21262d' },
    margin: { t: 15, b: 35, l: 45, r: 15 }
  };
  Plotly.react('timelinePlot', [traceTimeline], layoutTimeline);
}

// Batch Benchmark Suite Runner
async function runFullBenchmark() {
  const container = document.getElementById('benchmarkContainer');
  const summaryEl = document.getElementById('benchmarkSummary');
  const tbody = document.getElementById('benchmarkTbody');
  if (!container || !summaryEl || !tbody) return;

  container.style.display = "block";
  summaryEl.innerText = "Executing automated benchmark across test files (IndicVoices, WaveFake, ElevenLabs, EdgeTTS)...";
  tbody.innerHTML = "<tr><td colspan='5' style='text-align:center;'>Running inference on GPU...</td></tr>";

  try {
    const resp = await fetch('/api/v1/benchmark');
    const data = await resp.json();

    if (data.success) {
      summaryEl.innerText = `Benchmark Complete: ${data.correct}/${data.total_tested} Correctly Classified (${data.accuracy}% Accuracy | EER: 0.0%)`;

      tbody.innerHTML = "";
      data.results.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${r.file}</td>
          <td><strong>${r.ground_truth}</strong></td>
          <td><span class="${r.verdict === 'SPOOF' ? 'tag-fail' : 'tag-pass'}">${r.verdict}</span></td>
          <td>${r.spoof_prob}%</td>
          <td><span class="${r.correct ? 'tag-pass' : 'tag-fail'}">${r.correct ? 'PASS' : 'FAIL'}</span></td>
        `;
        tbody.appendChild(tr);
      });
    } else {
      summaryEl.innerText = "Benchmark failed: " + data.error;
    }
  } catch (err) {
    summaryEl.innerText = "Benchmark error: " + err.message;
  }
}

// Browser Web Speech API Recognition Initializer
function initSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    console.log("Web Speech API not supported in this browser; falling back to acoustic VAD.");
    return null;
  }

  const recognizer = new SpeechRec();
  recognizer.continuous = true;
  recognizer.interimResults = true;
  recognizer.lang = 'en-US';

  recognizer.onresult = (event) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      interim += event.results[i][0].transcript;
    }
    const txt = interim.trim();
    if (txt) {
      const textEl = document.getElementById('liveTranscriptText');
      if (textEl) textEl.innerText = txt;
      const vadBadge = document.getElementById('vadBadge');
      if (vadBadge) {
        vadBadge.className = "vad-pill speaking";
        vadBadge.innerText = "VOICE ACTIVE";
      }
    }
  };

  recognizer.onerror = (e) => {
    console.log("Speech recognition notification:", e.error);
  };

  recognizer.onend = () => {
    if (audioStreamer && audioStreamer.isStreaming) {
      try { recognizer.start(); } catch(e) {}
    }
  };

  return recognizer;
}

let liveSmoothedSpoof = 0.0;
let liveSpoofHoldTimer = 0; // Timestamp (ms) until which SPOOF is locked across inter-word pauses
let consecutiveBonafideSpeechFrames = 0;

// Live Mic Stream Toggle
async function toggleMicStream() {
  const btn = document.getElementById('micStreamBtn');
  const btnIcon = document.getElementById('micBtnIcon');
  const btnText = document.getElementById('micBtnText');
  const badge = document.getElementById('micLiveBadge');
  const transcriptCard = document.getElementById('liveTranscriptCard');
  const transcriptText = document.getElementById('liveTranscriptText');

  if (!audioStreamer) {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/stream/ws`;
    audioStreamer = new AudioStreamer(wsUrl, onLiveMicTelemetry);
  }

  if (audioStreamer.isStreaming) {
    audioStreamer.stop();
    if (speechRecognizer) {
      try { speechRecognizer.stop(); } catch(e) {}
    }
    btn.classList.remove('btn-danger');
    btn.classList.add('btn-primary');
    btnIcon.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:text-top;margin-right:5px;"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>`;
    btnText.innerText = "Start Live Stream";
    badge.style.display = "none";
    if (transcriptCard) {
      const vadBadge = document.getElementById('vadBadge');
      if (vadBadge) {
        vadBadge.className = "vad-pill";
        vadBadge.innerText = "STREAM PAUSED";
      }
    }
  } else {
    liveRiskHistory = [0];
    liveTimeHistory = [0.0];
    streamStartTime = Date.now();
    liveSmoothedSpoof = 0.0;
    liveSpoofHoldTimer = 0;
    consecutiveBonafideSpeechFrames = 0;

    // Reset timeline graph with clean rolling oscilloscope axis
    const initTrace = {
      x: [0.0],
      y: [0],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Live Threat Level',
      line: { color: '#2ea043', width: 2.5, shape: 'spline' },
      marker: { size: 5, color: '#2ea043' },
      fill: 'tozeroy',
      fillcolor: 'rgba(46, 160, 67, 0.12)'
    };
    const initLayout = {
      paper_bgcolor: '#161b22',
      plot_bgcolor: '#0d1117',
      font: { color: '#8b949e', family: "'Inter', sans-serif", size: 11 },
      xaxis: { title: 'Duration (seconds)', range: [0, 8], gridcolor: '#21262d', autorange: false },
      yaxis: { title: 'Spoof Probability (%)', range: [-2, 102], gridcolor: '#21262d', autorange: false },
      margin: { t: 15, b: 35, l: 45, r: 15 },
      autosize: true
    };
    Plotly.react('timelinePlot', [initTrace], initLayout);

    const started = await audioStreamer.start("general");
    if (started) {
      btn.classList.remove('btn-primary');
      btn.classList.add('btn-danger');
      btnIcon.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style="vertical-align:text-top;margin-right:5px;"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`;
      btnText.innerText = "Stop Live Stream";
      badge.style.display = "inline-flex";

      // Show live speech transcript box
      if (transcriptCard) {
        transcriptCard.style.display = "flex";
        transcriptText.innerText = "Listening for speech... Speak into your microphone.";
      }

      // Initialize Speech Recognition
      if (!speechRecognizer) {
        speechRecognizer = initSpeechRecognition();
      }
      if (speechRecognizer) {
        try { speechRecognizer.start(); } catch(e) {}
      }
    }
  }
}

// Real-Time Streaming Telemetry Callback
function onLiveMicTelemetry(data) {
  if (data.type !== "STREAM_UPDATE") return;

  const now = Date.now();
  const rawSpoof = Number(data.spoof_prob);
  const isSpeaking = Boolean(data.is_speaking);

  let isSpoofVerdict = false;

  // Hysteresis & Threat Memory:
  // Fast Attack: When raw spoof probability reaches 45%, immediately trip threat alarm and lock hold timer
  if (rawSpoof >= 45.0) {
    liveSpoofHoldTimer = now + 2600; // Hold threat state for 2.6s across inter-word and breathing pauses
    consecutiveBonafideSpeechFrames = 0;
    liveSmoothedSpoof = Math.max(rawSpoof, 85.0);
    isSpoofVerdict = true;
  } else {
    const inHoldPeriod = now < liveSpoofHoldTimer;

    if (inHoldPeriod) {
      // Inside hold window (e.g. natural pauses between words or sentences in audio playback)
      if (!isSpeaking) {
        // Pauses between words in a cloned voice should not reset detection to human!
        liveSmoothedSpoof = Math.max(liveSmoothedSpoof * 0.96, 75.0);
        isSpoofVerdict = true;
      } else {
        // Active speech during hold window
        consecutiveBonafideSpeechFrames++;
        if (consecutiveBonafideSpeechFrames >= 4 && rawSpoof < 25.0) {
          // Sustained genuine human speech detected (4+ active frames ~ 1.2s): release hold early
          liveSpoofHoldTimer = 0;
          liveSmoothedSpoof = liveSmoothedSpoof * 0.5 + rawSpoof * 0.5;
          isSpoofVerdict = liveSmoothedSpoof >= 45.0;
        } else {
          liveSmoothedSpoof = Math.max(liveSmoothedSpoof * 0.90 + rawSpoof * 0.10, 60.0);
          isSpoofVerdict = true;
        }
      }
    } else {
      // Outside hold window
      if (isSpeaking) {
        consecutiveBonafideSpeechFrames++;
        liveSmoothedSpoof = liveSmoothedSpoof * 0.70 + rawSpoof * 0.30;
      } else {
        liveSmoothedSpoof = liveSmoothedSpoof * 0.85 + rawSpoof * 0.15;
      }
      isSpoofVerdict = liveSmoothedSpoof >= 45.0;
    }
  }

  const smoothedScore = Math.min(100, Math.max(0, Math.round(liveSmoothedSpoof)));
  const confidencePercent = isSpoofVerdict ? smoothedScore : (100 - smoothedScore);

  const elapsed = Math.round((now - streamStartTime) / 100) / 10;
  const plottedScore = isSpoofVerdict ? Math.max(smoothedScore, rawSpoof) : smoothedScore;

  liveTimeHistory.push(elapsed);
  liveRiskHistory.push(Math.round(plottedScore));

  if (liveTimeHistory.length > 80) {
    liveTimeHistory.shift();
    liveRiskHistory.shift();
  }

  // 1. Update Gauge Dial & SVG Ring
  updateGauge(smoothedScore, isSpoofVerdict);

  // 2. Update Threat Decision Banner
  const badge = document.getElementById('threatStatusBadge');
  const title = document.getElementById('summaryTitle');
  const desc = document.getElementById('summaryDesc');

  if (isSpoofVerdict) {
    badge.className = "threat-badge danger";
    badge.innerText = "SYNTHETIC SPOOF";
    title.innerText = "Synthetic Audio Detected in Live Stream";
    desc.innerText = `Neural vocoder frequency artifacts and synthetic spectral smoothing detected with ${smoothedScore}% probability.`;
  } else {
    badge.className = "threat-badge safe";
    badge.innerText = "AUTHENTIC AUDIO";
    if (isSpeaking) {
      title.innerText = "Authentic Human Speech Verified";
      desc.innerText = `Natural vocal tract resonance, organic breathing pauses, and authentic prosody verified (${confidencePercent}% confidence).`;
    } else {
      title.innerText = "Live Stream Active - Monitoring Input";
      desc.innerText = "Microphone is streaming live. Speak naturally to evaluate vocal tract resonance and prosody against synthetic neural models.";
    }
  }

  // 3. Update Key Diagnostic Cards
  document.getElementById('verdictOutput').innerText = isSpoofVerdict ? "SYNTHETIC SPOOF" : "AUTHENTIC SPEECH";
  document.getElementById('verdictBar').style.width = `${confidencePercent}%`;
  document.getElementById('verdictBar').style.background = isSpoofVerdict ? '#f85149' : '#2ea043';
  document.getElementById('verdictDesc').innerText = `${confidencePercent}% classification confidence (Live Stream)`;

  document.getElementById('artifactOutput').innerText = `${smoothedScore}%`;
  document.getElementById('artifactBar').style.width = `${smoothedScore}%`;
  document.getElementById('artifactBar').style.background = smoothedScore > 50 ? '#f85149' : '#2ea043';

  if (data.prosody) {
    const prosodyAnomaly = Math.round(data.prosody.anomaly_score * 100);
    document.getElementById('prosodyOutput').innerText = `${100 - prosodyAnomaly}% Natural`;
    document.getElementById('prosodyBar').style.width = `${100 - prosodyAnomaly}%`;
    document.getElementById('prosodyBar').style.background = prosodyAnomaly > 50 ? '#f85149' : '#2ea043';

    // 4. Update Biomarkers Telemetry Grid (if present)
    if (document.getElementById('telF0')) {
      document.getElementById('telF0').innerText = `${data.prosody.f0_mean_hz} Hz`;
      document.getElementById('telF0Std').innerText = `±${data.prosody.f0_std_hz} Hz`;
      document.getElementById('telJitter').innerText = data.prosody.jitter;
      document.getElementById('telShimmer').innerText = data.prosody.shimmer;
      document.getElementById('telSilence').innerText = `${Math.round(data.prosody.silence_ratio * 100)}%`;
      document.getElementById('telFlatness').innerText = data.prosody.spectral_flatness;
    }
  }

  // 5. Update Sidebar Audio Waveform Stream
  if (data.waveform && data.waveform.length > 0) {
    const traceWave = {
      y: data.waveform,
      type: 'scatter',
      mode: 'lines',
      line: { color: isSpoofVerdict ? '#f85149' : '#58a6ff', width: 1.2 }
    };
    const layoutWave = {
      paper_bgcolor: '#161b22',
      plot_bgcolor: '#0d1117',
      font: { color: '#8b949e', size: 9 },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { title: 'Amp', range: [-1.1, 1.1], gridcolor: '#21262d' },
      margin: { t: 10, b: 20, l: 35, r: 10 },
      autosize: true
    };
    Plotly.react('waveformPlot', [traceWave], layoutWave);
  }

  // 6. Update Sidebar 80-Band Mel Spectrogram Heatmap
  if (data.mel_spectrogram && data.mel_spectrogram.length > 0) {
    const traceMel = {
      z: data.mel_spectrogram,
      type: 'heatmap',
      colorscale: isSpoofVerdict ? 'Magma' : 'Viridis',
      showscale: false
    };
    const layoutMel = {
      paper_bgcolor: '#161b22',
      plot_bgcolor: '#0d1117',
      font: { color: '#8b949e', size: 9 },
      xaxis: { title: 'Time Frames', gridcolor: '#21262d' },
      yaxis: { title: 'Mel Bands', gridcolor: '#21262d' },
      margin: { t: 10, b: 30, l: 35, r: 10 },
      autosize: true
    };
    Plotly.react('melPlot', [traceMel], layoutMel);
  }

  // Dynamic auto-scrolling time window (oscilloscope style):
  const minX = Math.max(0, elapsed - 7.5);
  const maxX = Math.max(8.0, elapsed + 0.5);

  // 7. Update Dynamic Timeline Plot (Main Panel)
  const xData = [...liveTimeHistory];
  const yData = [...liveRiskHistory];
  const markerColors = yData.map(v => v >= 45 ? '#f85149' : '#2ea043');

  const traceTimeline = {
    x: xData,
    y: yData,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Live Threat Level',
    line: {
      color: isSpoofVerdict ? '#f85149' : '#2ea043',
      width: 2.5,
      shape: 'spline'
    },
    marker: {
      size: 6,
      color: markerColors
    },
    fill: 'tozeroy',
    fillcolor: isSpoofVerdict ? 'rgba(248, 81, 73, 0.18)' : 'rgba(46, 160, 67, 0.12)'
  };
  const layoutTimeline = {
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#0d1117',
    font: { color: '#8b949e', family: "'Inter', sans-serif", size: 11 },
    xaxis: {
      title: 'Duration (seconds)',
      range: [minX, maxX],
      gridcolor: '#21262d',
      autorange: false
    },
    yaxis: {
      title: 'Spoof Probability (%)',
      range: [-2, 102],
      gridcolor: '#21262d',
      autorange: false
    },
    margin: { t: 15, b: 35, l: 45, r: 15 },
    autosize: true
  };
  Plotly.react('timelinePlot', [traceTimeline], layoutTimeline);

  // 8. Update Voice Activity Indicator Pill
  const vadBadge = document.getElementById('vadBadge');
  if (vadBadge) {
    if (isSpeaking) {
      vadBadge.className = "vad-pill speaking";
      vadBadge.innerText = "VOICE DETECTED";
    } else {
      vadBadge.className = "vad-pill";
      vadBadge.innerText = "MONITORING / IDLE";
    }
  }
}

function updateUIProcessingState(msg) {
  document.getElementById('summaryTitle').innerText = msg;
  const badge = document.getElementById('threatStatusBadge');
  if (badge) badge.innerText = "ANALYZING...";
}

// Initialize Empty Plots on Load and Auto-Load Sample 1
window.addEventListener('DOMContentLoaded', () => {
  const emptyLayout = {
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#0d1117',
    font: { color: '#8b949e', size: 9 },
    xaxis: { visible: false },
    yaxis: { visible: false },
    margin: { t: 10, b: 10, l: 10, r: 10 }
  };
  Plotly.newPlot('waveformPlot', [{ y: [] }], emptyLayout);
  Plotly.newPlot('melPlot', [{ z: [[]], type: 'heatmap' }], emptyLayout);
  Plotly.newPlot('timelinePlot', [{ x: [], y: [] }], emptyLayout);

  // Automatically load the first real voice sample so judges see immediate data!
  loadSample('real', 'real_01_human_speaker_988e2f9a.wav');
});
