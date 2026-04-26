import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import json, os, tempfile, time

st.set_page_config(
    page_title="SONIQ — Music Genre AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CONSTANTS
# ============================================================

GENRES = ['blues','classical','country','disco','hiphop',
          'jazz','metal','pop','reggae','rock']

GENRE_COLORS = {
    'blues':    '#4488ff', 'classical':'#cc77ff', 'country': '#ff9944',
    'disco':    '#ff44cc', 'hiphop':   '#00ddff', 'jazz':    '#ffdd00',
    'metal':    '#aaaaaa', 'pop':      '#ff6644', 'reggae':  '#44dd88',
    'rock':     '#ff4455',
}
GENRE_EMOJI = {
    'blues':'🎸','classical':'🎻','country':'🤠','disco':'🪩',
    'hiphop':'🎤','jazz':'🎷','metal':'🤘','pop':'⭐','reggae':'🌿','rock':'🔥',
}
GENRE_BPM = {
    'blues':'60–100','classical':'40–180','country':'80–130','disco':'110–135',
    'hiphop':'80–100','jazz':'100–200','metal':'100–200','pop':'100–130',
    'reggae':'60–90','rock':'110–140',
}
GENRE_INFO = {
    'blues':    'Blue notes, call-and-response patterns, 12-bar progressions.',
    'classical':'Orchestral instruments, complex harmonics, wide dynamic range.',
    'country':  'Acoustic guitar, pedal steel, Southern American storytelling.',
    'disco':    'Four-on-the-floor beat, funky basslines, orchestral strings.',
    'hiphop':   'Boom bap beats, sampling culture, heavy bass, rap vocals.',
    'jazz':     'Complex harmonies, improvisation, syncopation, swing feel.',
    'metal':    'Heavy distorted guitars, aggressive drumming, high tempo.',
    'pop':      'Catchy melodies, verse-chorus structure, accessible production.',
    'reggae':   'Offbeat rhythms (skank), deep bass, Jamaican roots.',
    'rock':     'Electric guitars, strong backbeat, energetic performance.',
}

SAMPLE_RATE    = 22050
SEGMENT_LENGTH = 3 * SAMPLE_RATE
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DARK_BG        = '#030308'
CARD_BG        = '#07071a'
GRID_COL       = '#111130'
TEXT_COL       = '#888899'



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ---- RESET & BASE ---- */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[class*="css"],
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.main, section.main {
    background-color: #030308 !important;
    color: #e0e0e0;
    font-family: 'Rajdhani', sans-serif;
}

/* Force dark background on every Streamlit wrapper */
[data-testid="stHeader"]    { background: transparent !important; }
[data-testid="stToolbar"]   { display: none !important; }
[data-testid="stDecoration"]{ display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
#MainMenu, footer, header   { visibility: hidden !important; }

.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px !important;
    background: transparent !important;
}

/* ---- HERO ---- */
.hero-wrap {
    position: relative; text-align: center;
    padding: 3rem 0 2rem; overflow: hidden;
}
.hero-glow {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse, rgba(100,70,255,.18) 0%, transparent 70%);
    animation: pulse-glow 4s ease-in-out infinite;
    pointer-events: none;
}
.hero-title {
    position: relative;
    font-family: 'Orbitron', monospace;
    font-size: 5rem; font-weight: 900; letter-spacing: .2em;
    background: linear-gradient(135deg,#6444ff 0%,#a259ff 40%,#ff59d6 70%,#ff9a3c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; background-size: 200% 200%;
    animation: shimmer 6s ease-in-out infinite;
}
.hero-sub {
    position: relative;
    font-family: 'Share Tech Mono', monospace;
    font-size: .85rem; color: #5555aa; letter-spacing: .25em;
    text-transform: uppercase; margin-top: .5rem;
    animation: blink-fade 3s ease-in-out infinite;
}

/* ---- KEYFRAMES ---- */
@keyframes shimmer {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
@keyframes blink-fade { 0%,100%{opacity:.7;} 50%{opacity:1;} }
@keyframes pulse-glow {
    0%,100% { opacity:.6; transform:translate(-50%,-50%) scale(1); }
    50%      { opacity:1;  transform:translate(-50%,-50%) scale(1.1); }
}
@keyframes scan-line-h { 0%{left:-100%} 100%{left:100%} }
@keyframes scan-line-v { 0%{transform:translateY(-100%)} 100%{transform:translateY(100%)} }
@keyframes nebula-drift {
    0%   { transform: translate(0,0) scale(1); }
    100% { transform: translate(-4%,3%) scale(1.1); }
}
@keyframes grid-pulse {
    0%,100% { opacity:.6 }
    50%      { opacity:1  }
}
@keyframes glitch {
    0%,95%,100% { text-shadow:none; transform:none; }
    96% { text-shadow:-3px 0 #ff0080,3px 0 #00ffff; transform:skewX(-2deg); }
    97% { text-shadow:3px 0 #ff0080,-3px 0 #00ffff;  transform:skewX(2deg); }
    98% { text-shadow:none; transform:none; }
}
@keyframes boot-bar { from{width:0} }
@keyframes float-particle {
    0%   { transform: translateY(0)   translateX(0); opacity:.3; }
    50%  { transform: translateY(-20px) translateX(10px); opacity:.7; }
    100% { transform: translateY(0)   translateX(0); opacity:.3; }
}

/* ---- STAT ROW ---- */
.stat-row { display:flex; gap:1rem; margin:2rem 0; }
.stat-card {
    flex:1; position:relative; overflow:hidden;
    background: linear-gradient(135deg,rgba(10,10,26,.9),rgba(15,15,34,.9));
    border:1px solid #1a1a3a; border-radius:16px;
    padding:1.2rem 1.5rem; text-align:center;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.stat-card-scan {
    position:absolute; top:0; left:-100%; right:0; height:2px;
    background: linear-gradient(90deg, transparent, #6444ff, transparent);
    animation: scan-line-h 3s ease-in-out infinite;
}
.stat-val { font-family:'Orbitron',monospace; font-size:2rem; font-weight:700; color:#8866ff; }
.stat-lbl { font-family:'Share Tech Mono',monospace; font-size:.7rem; color:#444488;
            letter-spacing:.2em; text-transform:uppercase; margin-top:4px; }

/* ---- UPLOAD ---- */
div[data-testid="stFileUploader"] {
    background: linear-gradient(135deg,#060614,#0a0a20) !important;
    border: 2px dashed #2a2a5a !important;
    border-radius: 20px !important; padding: 20px !important;
}

/* ---- RESULT HERO ---- */
.result-hero {
    position: relative; border-radius: 24px;
    padding: 3rem 2rem; text-align: center; margin: 2rem 0;
    overflow: hidden;
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
}
/* Holographic scanline sweep */
.result-hero-scan {
    position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(to bottom,
        rgba(255,255,255,0) 0%, rgba(255,255,255,.04) 50%, rgba(255,255,255,0) 100%);
    animation: scan-line-v 6s linear infinite;
    z-index: 1;
}
.genre-label {
    position: relative; z-index: 2;
    font-family: 'Orbitron', monospace;
    font-size: 4rem; font-weight: 900; letter-spacing: .15em; text-transform: uppercase;
    animation: glitch 8s infinite;
}
.confidence-display {
    position: relative; z-index: 2;
    font-family:'Share Tech Mono',monospace; font-size:1.2rem;
    color:#888; letter-spacing:.1em; margin:.5rem 0;
}

/* ---- MODEL CARDS ---- */
.model-card {
    background: rgba(7,7,26,.9);
    border-radius: 18px; padding: 1.5rem; text-align: center;
    border: 1px solid #1a1a33;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    transition: transform .2s, border-color .3s;
}
.model-card:hover { transform: translateY(-3px); }
.model-card-title { font-family:'Share Tech Mono',monospace; font-size:.7rem; color:#444488; letter-spacing:.2em; text-transform:uppercase; margin-bottom:.75rem; }
.model-prediction  { font-family:'Orbitron',monospace; font-size:1.3rem; font-weight:700; letter-spacing:.1em; }
.model-conf        { font-family:'Share Tech Mono',monospace; font-size:1rem; color:#666; margin:4px 0; }
.agree-badge       { display:inline-block; padding:3px 12px; border-radius:50px; font-size:.72rem;
                     font-family:'Share Tech Mono',monospace; letter-spacing:.1em; margin-top:8px; }
.model-accuracy    { font-family:'Share Tech Mono',monospace; font-size:.65rem; color:#333366; margin-top:8px; letter-spacing:.1em; }

/* ---- SECTION LABELS ---- */
.section-label {
    font-family:'Share Tech Mono',monospace; font-size:.72rem; color:#444488;
    letter-spacing:.25em; text-transform:uppercase; margin:2rem 0 .75rem;
    display:flex; align-items:center; gap:12px;
}
.section-label::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#1a1a3a,transparent); }

/* ---- DNA BARS ---- */
.dna-row { display:flex; align-items:center; gap:12px; margin:8px 0; }
.dna-label { font-family:'Share Tech Mono',monospace; font-size:.68rem; color:#444488; letter-spacing:.1em; width:140px; flex-shrink:0; }
.dna-bar-wrap { flex:1; height:8px; background:#0d0d22; border-radius:4px; overflow:hidden; }
.dna-bar  { height:100%; border-radius:4px; }
.dna-val  { font-family:'Share Tech Mono',monospace; font-size:.68rem; color:#555588; width:80px; text-align:right; flex-shrink:0; }

/* ---- TOP 3 ---- */
.top3-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 16px; margin:6px 0;
    background: rgba(7,7,26,.9); border-radius:10px; border-left:3px solid transparent;
}
.top3-genre { font-family:'Rajdhani',sans-serif; font-weight:600; font-size:1rem; letter-spacing:.05em; }
.top3-pct   { font-family:'Share Tech Mono',monospace; font-size:.9rem; color:#888; }

/* ---- BOOT ---- */
.boot-wrap {
    text-align:center; padding:4rem 2rem;
    font-family:'Share Tech Mono',monospace;
    background: rgba(3,3,8,.95);
    border-radius: 24px;
    border: 1px solid #1a1a3a;
    margin: 2rem 0;
}
.boot-pct {
    font-family:'Orbitron',monospace; font-size:3rem;
    font-weight:900; color:#8866ff; margin:.5rem 0;
}
.boot-bar-outer {
    width:420px; height:6px; background:#0d0d22;
    border-radius:3px; margin:1rem auto; overflow:hidden;
}
.boot-bar-inner {
    height:100%; border-radius:3px;
    background: linear-gradient(90deg,#6444ff,#ff44cc);
    box-shadow: 0 0 12px #6444ff88;
}
.boot-label {
    font-size:.72rem; color:#333366; letter-spacing:.25em; margin-top:.5rem;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] { background:#07071a; border-radius:12px; padding:4px; gap:2px; }
.stTabs [data-baseweb="tab"]      { font-family:'Share Tech Mono',monospace; font-size:.75rem; letter-spacing:.1em; color:#444488; border-radius:8px; padding:8px 20px; }
.stTabs [aria-selected="true"]    { background:linear-gradient(135deg,#1a1a3a,#0f0f2e) !important; color:#8866ff !important; }

/* ---- MISC ---- */
div[data-baseweb="select"] { font-family:'Share Tech Mono',monospace !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#030308; }
::-webkit-scrollbar-thumb { background:#2a2a5a; border-radius:2px; }
</style>
""", unsafe_allow_html=True)



st.markdown("""
<!-- Nebula glow layer -->
<div id="soniq-nebula" style="
    position:fixed; inset:-20%; z-index:-5; pointer-events:none;
    background:
        radial-gradient(circle at 25% 30%, rgba(110,80,255,.18), transparent 35%),
        radial-gradient(circle at 75% 70%, rgba(255,70,220,.14), transparent 40%);
    animation: nebula-drift 22s ease-in-out infinite alternate;
"></div>

<!-- Grid depth layer -->
<div id="soniq-grid" style="
    position:fixed; inset:0; z-index:-4; pointer-events:none;
    background-image:
        linear-gradient(rgba(120,90,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(120,90,255,.04) 1px, transparent 1px);
    background-size: 45px 45px;
    animation: grid-pulse 8s ease-in-out infinite;
"></div>

<!-- Genre reactive tint layer (color injected dynamically after prediction) -->
<div id="soniq-genre-bg" style="
    position:fixed; inset:0; z-index:-3; pointer-events:none;
    opacity:0; transition: opacity 1.2s ease;
"></div>
""", unsafe_allow_html=True)



components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin:0; padding:0; }
  html, body { background:transparent; overflow:hidden; width:1px; height:1px; }
</style>
</head>
<body>
<script>
(function() {
    // Wait for parent DOM to be ready
    function inject() {
        try {
            const P = window.parent;
            const doc = P.document;

            // Avoid duplicate canvas on Streamlit re-runs
            if (doc.getElementById('soniq-canvas')) return;

            const c = doc.createElement('canvas');
            c.id = 'soniq-canvas';
            Object.assign(c.style, {
                position: 'fixed', top: '0', left: '0',
                width: '100vw', height: '100vh',
                zIndex: '-6', pointerEvents: 'none',
                display: 'block'
            });
            doc.body.appendChild(c);

            const ctx = c.getContext('2d');
            let w, h, mx = P.innerWidth / 2, my = P.innerHeight / 2, t = 0;
            const particles = [];

            function resize() {
                w = c.width  = P.innerWidth;
                h = c.height = P.innerHeight;
            }
            P.addEventListener('resize', resize);
            resize();
            P.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });

            for (let i = 0; i < 100; i++) {
                particles.push({
                    x:  Math.random() * P.innerWidth,
                    y:  Math.random() * P.innerHeight,
                    r:  Math.random() * 2 + 0.4,
                    dx: (Math.random() - 0.5) * 0.4,
                    dy: (Math.random() - 0.5) * 0.4,
                });
            }

            function draw() {
                t += 0.012;
                ctx.clearRect(0, 0, w, h);

                // Sine wave grid lines
                ctx.lineWidth = 1;
                ctx.strokeStyle = 'rgba(120,90,255,.06)';
                for (let row = 0; row < h; row += 44) {
                    ctx.beginPath();
                    for (let x = 0; x <= w; x += 18) {
                        const y = row + Math.sin(x * 0.012 + t) * 5;
                        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                    }
                    ctx.stroke();
                }

                // Mouse-reactive radial glow
                const g = ctx.createRadialGradient(mx, my, 0, mx, my, 260);
                g.addColorStop(0, 'rgba(120,80,255,.16)');
                g.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.fillStyle = g;
                ctx.fillRect(0, 0, w, h);

                // Particles
                particles.forEach(p => {
                    p.x += p.dx; p.y += p.dy;
                    if (p.x < 0 || p.x > w) p.dx *= -1;
                    if (p.y < 0 || p.y > h) p.dy *= -1;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(120,80,255,.4)';
                    ctx.fill();
                });

                requestAnimationFrame(draw);
            }
            draw();
        } catch(e) {
            console.warn('SONIQ canvas inject error:', e);
        }
    }

    // Retry a few times in case parent DOM isn't ready yet
    let tries = 0;
    const id = setInterval(() => {
        inject();
        if (++tries > 5) clearInterval(id);
    }, 300);
})();
</script>
</body>
</html>
""", height=0, scrolling=False)

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(.25),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(.25),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(.25),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Sequential(
            nn.Flatten(),nn.Linear(256,512),nn.BatchNorm1d(512),
            nn.ReLU(),nn.Dropout(.5),nn.Linear(512,num_classes),
        )
    def forward(self, x):
        return self.classifier(self.global_pool(self.features(x)))


class LSTMModel(nn.Module):
    def __init__(self, input_size=13, num_classes=10):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size,128,batch_first=True,bidirectional=True)
        self.drop1 = nn.Dropout(.3)
        self.lstm2 = nn.LSTM(256,64,batch_first=True,bidirectional=True)
        self.drop2 = nn.Dropout(.3)
        self.classifier = nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,num_classes),
        )
    def forward(self, x):
        out,_=self.lstm1(x); out=self.drop1(out)
        out,_=self.lstm2(out); out=self.drop2(out[:,-1,:])
        return self.classifier(out)


class HybridModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d((2,4)),nn.Dropout2d(.25),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d((2,4)),nn.Dropout2d(.25),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d((2,2)),
        )
        self.lstm = nn.LSTM(128*16,128,batch_first=True,bidirectional=True)
        self.drop = nn.Dropout(.3)
        self.classifier = nn.Sequential(
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,num_classes),
        )
    def forward(self, x):
        x=self.cnn(x); b,c,h,w=x.shape
        x=x.permute(0,3,1,2).reshape(b,w,c*h)
        out,_=self.lstm(x)
        return self.classifier(self.drop(out[:,-1,:]))


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_all_models():
    norm = np.load('data/processed/norm_stats.npy') \
           if os.path.exists('data/processed/norm_stats.npy') else None
    meta = {}
    if os.path.exists('data/processed/metadata.json'):
        with open('data/processed/metadata.json') as f:
            meta = json.load(f)
    mfcc_dim = meta.get('mfcc_shape', [0,0,13])[2]
    models = {}
    configs = [
        ('CNN',            CNNModel,    'results/CNN_best.pth',             False, 90.52),
        ('LSTM',           LSTMModel,   'results/LSTM_best.pth',            True,  81.51),
        ('Hybrid CNN-LSTM',HybridModel, 'results/Hybrid_CNN_LSTM_best.pth', False, 86.58),
    ]
    for name, cls, path, use_mfcc, acc in configs:
        if os.path.exists(path):
            kwargs = {'input_size': mfcc_dim} if use_mfcc else {}
            m = cls(**kwargs).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models[name] = {'model': m, 'use_mfcc': use_mfcc, 'acc': acc}
    return models, norm


# ============================================================
# AUDIO PROCESSING
# ============================================================

def process_audio(audio_bytes, norm_stats=None):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
        if len(y) > SEGMENT_LENGTH:
            mid = len(y) // 2
            seg = y[mid - SEGMENT_LENGTH//2 : mid + SEGMENT_LENGTH//2]
        else:
            seg = np.pad(y, (0, max(0, SEGMENT_LENGTH - len(y))))

        mel    = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - norm_stats[0]) / norm_stats[1] \
                   if norm_stats is not None \
                   else (mel_db - mel_db.mean()) / mel_db.std()
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13).T

        tempo_arr, _ = librosa.beat.beat_track(y=seg, sr=sr)
        tempo = float(tempo_arr) if np.ndim(tempo_arr) == 0 else float(tempo_arr[0])
        rms_frames = librosa.feature.rms(y=seg)[0]

        features = {
            'spectral_centroid':  float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr))),
            'spectral_rolloff':   float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(seg))),
            'rms_energy':         float(np.mean(rms_frames)),
            'rms_frames':         rms_frames,
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr))),
            'tempo':              tempo,
            'chroma':             librosa.feature.chroma_stft(y=seg, sr=sr),
        }
        return y, seg, sr, mel_db, mel_norm, mfcc, features
    finally:
        os.unlink(tmp_path)


def predict_all(models, mel_norm, mfcc):
    results = {}
    for name, info in models.items():
        with torch.no_grad():
            if info['use_mfcc']:
                x = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
            else:
                x = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(device)
            probs = torch.softmax(info['model'](x), dim=1).cpu().numpy()[0]
        top = np.argsort(probs)[::-1]
        results[name] = {
            'probs':     probs,
            'top_genre': GENRES[top[0]],
            'top_conf':  float(probs[top[0]]),
            'sorted':    [{'genre': GENRES[i], 'confidence': float(probs[i])} for i in top],
            'acc':       info['acc'],
        }
    return results


# ============================================================
# MATPLOTLIB PLOTS (unchanged analytics)
# ============================================================

NEON_CMAP   = LinearSegmentedColormap.from_list('neon',
    ['#000010','#0a0044','#220077','#6600cc','#cc00ff','#ff44ff','#ffaaff','#ffffff'])
MFCC_CMAP   = LinearSegmentedColormap.from_list('mfcc',
    ['#000020','#0a0050','#4400cc','#8844ff','#ff44ff','#ffccff'])
CHROMA_CMAP = LinearSegmentedColormap.from_list('chroma',
    ['#020210','#1a0044','#6444ff','#ff44ff'])


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(CARD_BG)
    ax.spines['bottom'].set_color('#1a1a3a'); ax.spines['left'].set_color('#1a1a3a')
    ax.spines['top'].set_visible(False);      ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8, labelpad=6, fontfamily='monospace')
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8, labelpad=6, fontfamily='monospace')
    ax.set_title(title,   color='#aaaacc', fontsize=10, pad=10, fontfamily='monospace', fontweight='bold')
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.7)


def plot_audio_dashboard(y_full, seg, sr, mel_db, features):
    fig = plt.figure(figsize=(16,9), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3,3, figure=fig, hspace=0.5, wspace=0.4)

    ax1 = fig.add_subplot(gs[0,:])
    ax1.set_facecolor(CARD_BG)
    t = np.linspace(0, len(y_full)/sr, len(y_full))
    ax1.plot(t, y_full, color='#4444aa', lw=0.4, alpha=0.5)
    ax1.fill_between(t, y_full, alpha=0.15, color='#6444ff')
    mid_sec = len(y_full)/sr/2
    ax1.axvspan(mid_sec-1.5, mid_sec+1.5, alpha=0.18, color='#a259ff', zorder=3)
    ax1.axvline(mid_sec-1.5, color='#a259ff', lw=1, alpha=0.6)
    ax1.axvline(mid_sec+1.5, color='#a259ff', lw=1, alpha=0.6)
    ax1.text(mid_sec, ax1.get_ylim()[1]*0.75, 'ANALYZED SEGMENT',
             ha='center', va='center', color='#a259ff', fontsize=7, fontfamily='monospace')
    style_ax(ax1, 'WAVEFORM — Full Track', 'Time (s)', 'Amplitude')

    ax2 = fig.add_subplot(gs[1:,:2])
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel',
                                    sr=sr, hop_length=HOP_LENGTH, ax=ax2, cmap=NEON_CMAP)
    cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB', pad=0.01)
    cbar.ax.tick_params(colors=TEXT_COL, labelsize=7)
    style_ax(ax2, 'MEL SPECTROGRAM — CNN Input (3-sec segment)', 'Time', 'Frequency (Hz)')

    ax3 = fig.add_subplot(gs[1,2])
    librosa.display.specshow(features['chroma'], x_axis='time', y_axis='chroma',
                              ax=ax3, cmap=CHROMA_CMAP, sr=sr)
    style_ax(ax3, 'CHROMAGRAM — Musical Key', 'Time', 'Pitch')

    ax4 = fig.add_subplot(gs[2,2])
    rms_frames = features['rms_frames']
    t_rms = np.linspace(0, 3, len(rms_frames))
    ax4.plot(t_rms, rms_frames, color='#ff44cc', lw=1.5)
    ax4.fill_between(t_rms, rms_frames, alpha=0.3, color='#ff44cc')
    style_ax(ax4, 'RMS ENERGY — Loudness', 'Time (s)', 'RMS')

    plt.suptitle('AUDIO ANALYSIS DASHBOARD', color='#3333aa',
                 fontsize=9, fontfamily='monospace', y=0.02, alpha=0.5)
    return fig


def plot_probability_chart(all_preds, model_name):
    preds  = all_preds[model_name]['sorted']
    genres = [p['genre'].upper() for p in preds]
    confs  = [p['confidence']*100 for p in preds]
    colors = [GENRE_COLORS[p['genre']] for p in preds]

    fig, ax = plt.subplots(figsize=(10,5.5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    y_pos = np.arange(len(genres))
    ax.barh(y_pos, [100]*len(genres), color='#0d0d22', height=0.55, zorder=1)
    bars = ax.barh(y_pos, confs, color=colors, height=0.55, alpha=0.9, zorder=2)
    ax.barh(y_pos, confs, color=colors, height=0.72, alpha=0.12, zorder=1)
    max_conf = max(confs)
    for bar, conf, col in zip(bars, confs, colors):
        ax.text(min(conf+1,97), bar.get_y()+bar.get_height()/2,
                f'{conf:.1f}%', va='center', color='white', fontsize=10,
                fontweight='700', fontfamily='monospace')
        if conf == max_conf:
            ax.text(1, bar.get_y()+bar.get_height()/2, '◀ TOP',
                    va='center', color=col, fontsize=7, fontfamily='monospace', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genres, color='#888899', fontsize=9, fontfamily='monospace')
    ax.set_xlim([0,112]); ax.invert_yaxis()
    ax.set_xlabel('Confidence %', color=TEXT_COL, fontsize=8, fontfamily='monospace')
    ax.set_title(f'{model_name.upper()} — GENRE PROBABILITIES',
                 color='#aaaacc', fontsize=10, fontfamily='monospace', fontweight='bold', pad=12)
    ax.spines['bottom'].set_color('#1a1a3a'); ax.spines['left'].set_color('#1a1a3a')
    ax.spines['top'].set_visible(False);      ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COL)
    ax.grid(axis='x', color=GRID_COL, linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    return fig


def plot_radar(all_preds):
    best   = max(all_preds, key=lambda k: all_preds[k]['top_conf'])
    genres = [p['genre'] for p in all_preds[best]['sorted'][:8]]
    angles = np.linspace(0, 2*np.pi, len(genres), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True), facecolor=DARK_BG)
    ax.set_facecolor('#06061a')
    ax.spines['polar'].set_color('#1a1a3a')
    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [g.upper() for g in genres],
                      color='#777799', fontsize=8, fontfamily='monospace')
    ax.set_yticklabels([]); ax.set_ylim(0,1)
    ax.grid(color='#111130', linewidth=0.8)
    colors_m = {'CNN':'#6444ff','LSTM':'#ff44cc','Hybrid CNN-LSTM':'#44ffaa'}
    for mname, mdata in all_preds.items():
        vals = [mdata['probs'][GENRES.index(g)] for g in genres] + \
               [mdata['probs'][GENRES.index(genres[0])]]
        col  = colors_m.get(mname,'#fff')
        ax.plot(angles, vals, color=col, linewidth=2, label=mname)
        ax.fill(angles, vals, color=col, alpha=0.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45,1.15),
              framealpha=0, labelcolor='#999', fontsize=8,
              prop={'family':'monospace','size':8})
    ax.set_title('MODEL AGREEMENT\nRADAR', color='#3333aa',
                 fontsize=9, fontfamily='monospace', pad=20)
    plt.tight_layout()
    return fig


def plot_mfcc_heatmap(mfcc):
    fig, ax = plt.subplots(figsize=(10,3.5), facecolor=DARK_BG)
    im = ax.imshow(mfcc.T, aspect='auto', origin='lower', cmap=MFCC_CMAP, interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_COL, labelsize=7)
    style_ax(ax, 'MFCC HEATMAP — LSTM Temporal Input', 'Time Frame', 'MFCC Coefficient')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN UI — HERO + STATS
# ============================================================

st.markdown("""
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <div class="hero-title">SONIQ</div>
    <div class="hero-sub">▸ Neural Music Genre Classification System ◂</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-row">
    <div class="stat-card"><div class="stat-card-scan"></div>
        <div class="stat-val">3</div><div class="stat-lbl">Neural Models</div></div>
    <div class="stat-card"><div class="stat-card-scan"></div>
        <div class="stat-val">10</div><div class="stat-lbl">Genres</div></div>
    <div class="stat-card"><div class="stat-card-scan"></div>
        <div class="stat-val">9,981</div><div class="stat-lbl">Training Segments</div></div>
    <div class="stat-card"><div class="stat-card-scan"></div>
        <div class="stat-val">90.5%</div><div class="stat-lbl">CNN Accuracy</div></div>
    <div class="stat-card"><div class="stat-card-scan"></div>
        <div class="stat-val">86.6%</div><div class="stat-lbl">Hybrid Accuracy</div></div>
    <div class="stat-card"><div class="stat-card-scan"></div>
        <div class="stat-val">{str(device).upper()}</div><div class="stat-lbl">Device</div></div>
</div>
""", unsafe_allow_html=True)

models, norm_stats = load_all_models()
if not models:
    st.error("No trained models found in results/. Run python step3_train.py first.")
    st.stop()

ctrl1, ctrl2 = st.columns([2,1])
with ctrl1:
    st.markdown('<div class="section-label">UPLOAD AUDIO FILE</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=['mp3','wav','ogg','flac'], label_visibility="collapsed")
with ctrl2:
    st.markdown('<div class="section-label">PRIMARY MODEL</div>', unsafe_allow_html=True)
    active_model = st.selectbox("", list(models.keys()), label_visibility="collapsed")

# ============================================================
# RESULTS BLOCK
# ============================================================

if uploaded:
    st.audio(uploaded)

    
    boot_slot = st.empty()
    boot_steps = [
        (8,  "LOADING AUDIO BUFFER"),
        (20, "RESAMPLING TO 22050 HZ"),
        (34, "EXTRACTING MEL SPECTROGRAM"),
        (48, "COMPUTING 13 MFCC COEFFICIENTS"),
        (58, "NORMALISING SIGNAL"),
        (68, "INFERENCE — CNN MODEL"),
        (78, "INFERENCE — LSTM MODEL"),
        (88, "INFERENCE — HYBRID CNN-LSTM"),
        (96, "FUSING PREDICTIONS"),
        (100,"ANALYSIS COMPLETE ◈"),
    ]
    for pct, label in boot_steps:
        boot_slot.markdown(f"""
        <div class="boot-wrap">
            <div style="font-size:.68rem;color:#5555aa;letter-spacing:.4em;margin-bottom:.75rem;">
                ◈ AI CORE INITIALISING ◈
            </div>
            <div class="boot-pct">{pct}%</div>
            <div class="boot-bar-outer">
                <div class="boot-bar-inner" style="width:{pct}%;"></div>
            </div>
            <div class="boot-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.06)
    boot_slot.empty()

    # Actual inference
    audio_bytes = uploaded.read()
    y_full, seg, sr, mel_db, mel_norm, mfcc_data, features = process_audio(audio_bytes, norm_stats)
    all_preds = predict_all(models, mel_norm, mfcc_data)

    top   = all_preds[active_model]
    color = GENRE_COLORS[top['top_genre']]
    emoji = GENRE_EMOJI[top['top_genre']]

   
    components.html(f"""
    <script>
    (function() {{
        try {{
            const bg = window.parent.document.getElementById('soniq-genre-bg');
            if (bg) {{
                bg.style.background = 'radial-gradient(circle at 30% 25%, {color}22, transparent 40%), radial-gradient(circle at 70% 75%, {color}18, transparent 40%)';
                bg.style.opacity = '1';
            }}
        }} catch(e) {{ console.warn('Genre lighting failed:', e); }}
    }})();
    </script>
    """, height=0, scrolling=False)

   
    rms_raw     = features['rms_energy']
    glow_px     = int(np.clip(rms_raw * 340, 20, 110))
    glow_alpha  = round(float(np.clip(rms_raw * 5.5, 0.08, 0.55)), 2)
    glow_hex    = hex(int(glow_alpha * 255))[2:].zfill(2)

    # Hero result card
    st.markdown(f"""
    <div class="result-hero" style="
        background: linear-gradient(135deg, {color}12 0%, {color}07 50%, #030308 100%);
        border: 1px solid {color}33;
        box-shadow: 0 0 {glow_px}px {color}{glow_hex},
                    0 0 {glow_px*2}px {color}{hex(max(int(glow_alpha*80),8))[2:].zfill(2)},
                    inset 0 0 {glow_px//2}px {color}0a;">
        <div class="result-hero-scan"></div>
        <div style="position:relative;z-index:2;font-family:monospace;font-size:.72rem;
                    color:{color}55;letter-spacing:.3em;text-transform:uppercase;margin-bottom:.5rem;">
            ◈ CLASSIFICATION RESULT ◈
        </div>
        <div style="position:relative;z-index:2;font-size:5rem;margin:.5rem 0;">{emoji}</div>
        <div class="genre-label" style="color:{color};text-shadow:0 0 {glow_px}px {color}88;">
            {top['top_genre'].upper()}
        </div>
        <div class="confidence-display">
            CONFIDENCE:&nbsp;
            <span style="color:{color};font-size:1.5rem;font-weight:700;">
                {top['top_conf']*100:.1f}%
            </span>
        </div>
        <div style="position:relative;z-index:2;margin:1rem auto;max-width:400px;height:6px;
                    background:#0a0a1a;border-radius:3px;overflow:hidden;">
            <div style="width:{top['top_conf']*100:.1f}%;height:100%;
                        background:linear-gradient(90deg,{color}88,{color});
                        border-radius:3px;box-shadow:0 0 14px {color};"></div>
        </div>
        <div style="position:relative;z-index:2;font-size:.9rem;color:#444466;max-width:500px;
                    margin:.5rem auto;line-height:1.8;font-family:monospace;">
            {GENRE_INFO.get(top['top_genre'],'')}
        </div>
        <div style="position:relative;z-index:2;margin-top:.75rem;font-family:monospace;
                    font-size:.72rem;color:#333366;">
            TYPICAL BPM: <span style="color:{color}88;">{GENRE_BPM.get(top['top_genre'],'N/A')}</span>
            &nbsp;·&nbsp;
            DETECTED TEMPO: <span style="color:{color}88;">{features['tempo']:.0f} BPM</span>
            &nbsp;·&nbsp;
            RMS: <span style="color:{color}88;">{rms_raw:.4f}</span>
            &nbsp;·&nbsp;
            GLOW: <span style="color:{color}88;">{glow_px}px</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- ALL MODELS ----
    st.markdown('<div class="section-label">ALL MODELS — SIMULTANEOUS PREDICTION</div>',
                unsafe_allow_html=True)
    mcols = st.columns(len(models))
    for col_w, (mname, mdata) in zip(mcols, all_preds.items()):
        mc    = GENRE_COLORS[mdata['top_genre']]
        em    = GENRE_EMOJI[mdata['top_genre']]
        agree = mdata['top_genre'] == top['top_genre']
        bc = '#44dd88' if agree else '#ffaa33'
        bb = '#44dd8822' if agree else '#ffaa3322'
        bt = '✓ AGREES' if agree else '≠ DIFFERS'
        with col_w:
            st.markdown(f"""
            <div class="model-card" style="border-color:{mc}33;box-shadow:0 0 20px {mc}0a;">
                <div class="model-card-title">{mname}</div>
                <div style="font-size:2rem;margin:4px 0;">{em}</div>
                <div class="model-prediction" style="color:{mc};">{mdata['top_genre'].upper()}</div>
                <div class="model-conf">{mdata['top_conf']*100:.1f}%</div>
                <div style="margin:6px 0;height:4px;background:#0d0d22;border-radius:2px;overflow:hidden;">
                    <div style="width:{mdata['top_conf']*100:.1f}%;height:100%;
                                background:{mc};box-shadow:0 0 6px {mc}88;"></div>
                </div>
                <span class="agree-badge"
                      style="background:{bb};color:{bc};border:1px solid {bc}44;">{bt}</span>
                <div class="model-accuracy">TEST ACC: {mdata['acc']}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- CHARTS ----
    cl, cr = st.columns([1.6,1])
    with cl:
        st.markdown('<div class="section-label">PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
        fig = plot_probability_chart(all_preds, active_model)
        st.pyplot(fig, use_container_width=True); plt.close()
    with cr:
        st.markdown('<div class="section-label">MODEL AGREEMENT RADAR</div>', unsafe_allow_html=True)
        fig = plot_radar(all_preds)
        st.pyplot(fig, use_container_width=True); plt.close()

    # ---- DNA BARS ----
    st.markdown('<div class="section-label">AUDIO FEATURE DNA</div>', unsafe_allow_html=True)
    def nf(v,lo,hi): return min(max((v-lo)/(hi-lo),0),1)
    dna = [
        ('SPECTRAL CENTROID',  nf(features['spectral_centroid'],  500,4000), f"{features['spectral_centroid']:.0f} Hz",  '#6444ff'),
        ('SPECTRAL ROLLOFF',   nf(features['spectral_rolloff'],  1000,8000), f"{features['spectral_rolloff']:.0f} Hz",   '#a259ff'),
        ('SPECTRAL BANDWIDTH', nf(features['spectral_bandwidth'], 500,3000), f"{features['spectral_bandwidth']:.0f} Hz", '#ff59d6'),
        ('ZERO CROSS RATE',    nf(features['zero_crossing_rate'],   0, 0.3), f"{features['zero_crossing_rate']:.4f}",    '#ff9a3c'),
        ('RMS ENERGY',         nf(features['rms_energy'],           0, 0.3), f"{features['rms_energy']:.4f}",            '#43e97b'),
        ('TEMPO (BPM)',        nf(features['tempo'],               40, 220), f"{features['tempo']:.1f} BPM",             '#00d2ff'),
    ]
    dc1, dc2 = st.columns(2)
    for idx, (label, val, display, col) in enumerate(dna):
        with (dc1 if idx < 3 else dc2):
            st.markdown(f"""
            <div class="dna-row">
                <div class="dna-label">{label}</div>
                <div class="dna-bar-wrap">
                    <div class="dna-bar" style="width:{val*100:.1f}%;
                        background:linear-gradient(90deg,{col}88,{col});
                        box-shadow:0 0 8px {col}66;"></div>
                </div>
                <div class="dna-val">{display}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- TOP 3 ----
    st.markdown('<div class="section-label">TOP-3 PREDICTIONS PER MODEL</div>', unsafe_allow_html=True)
    medals = ['🥇','🥈','🥉']
    t3cols = st.columns(len(models))
    for col_w, (mname, mdata) in zip(t3cols, all_preds.items()):
        with col_w:
            st.markdown(f'<div style="font-family:monospace;font-size:.7rem;color:#333366;'
                        f'letter-spacing:.15em;margin-bottom:8px;">{mname.upper()}</div>',
                        unsafe_allow_html=True)
            for j, p in enumerate(mdata['sorted'][:3]):
                mc  = GENRE_COLORS[p['genre']]
                pct = p['confidence']*100
                st.markdown(f"""
                <div class="top3-row" style="border-left-color:{mc};">
                    <span style="font-size:1rem;margin-right:8px;">{medals[j]}</span>
                    <span class="top3-genre" style="color:{mc};">{p['genre'].capitalize()}</span>
                    <span class="top3-pct">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

    # ---- AUDIO DASHBOARD ----
    st.markdown('<div class="section-label">AUDIO ANALYSIS DASHBOARD</div>', unsafe_allow_html=True)
    fig = plot_audio_dashboard(y_full, seg, sr, mel_db, features)
    st.pyplot(fig, use_container_width=True); plt.close()

    # ---- MFCC ----
    st.markdown('<div class="section-label">MFCC HEATMAP — LSTM TEMPORAL INPUT</div>', unsafe_allow_html=True)
    fig = plot_mfcc_heatmap(mfcc_data)
    st.pyplot(fig, use_container_width=True); plt.close()

    # ---- FULL BREAKDOWN TABS ----
    st.markdown('<div class="section-label">FULL PROBABILITY BREAKDOWN</div>', unsafe_allow_html=True)
    tabs = st.tabs([f"⬡ {m}" for m in all_preds.keys()])
    for tab, (mname, mdata) in zip(tabs, all_preds.items()):
        with tab:
            tc1, tc2 = st.columns([1,1.5])
            with tc1:
                for p in mdata['sorted']:
                    mc  = GENRE_COLORS[p['genre']]
                    pct = p['confidence']*100
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;
                                padding:7px 0;border-bottom:1px solid #0d0d22;">
                        <span style="width:12px;height:12px;border-radius:50%;background:{mc};
                                     box-shadow:0 0 6px {mc};flex-shrink:0;display:inline-block;"></span>
                        <span style="font-family:monospace;font-size:.85rem;color:{mc};flex:1;">
                            {p['genre'].upper()}</span>
                        <span style="font-family:monospace;font-size:.85rem;color:#555577;
                                     width:50px;text-align:right;">{pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            with tc2:
                fig = plot_probability_chart(all_preds, mname)
                st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown(f"""
    <div style="text-align:center;margin-top:3rem;padding:1.5rem;
                border-top:1px solid #0d0d22;font-family:monospace;
                font-size:.65rem;color:#222244;letter-spacing:.2em;">
        SONIQ &nbsp;·&nbsp; PYTORCH {torch.__version__} &nbsp;·&nbsp; {str(device).upper()}
        &nbsp;·&nbsp; CNN 90.52% &nbsp;·&nbsp; LSTM 81.51% &nbsp;·&nbsp; HYBRID 86.58%
        &nbsp;·&nbsp; GTZAN
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# EMPTY STATE
# ============================================================

else:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;">
        <div style="font-size:5rem;margin-bottom:1.5rem;">🎵</div>
        <div style="font-family:monospace;font-size:.9rem;color:#333366;
                    letter-spacing:.3em;text-transform:uppercase;margin-bottom:.5rem;">
            UPLOAD AUDIO TO BEGIN ANALYSIS
        </div>
        <div style="font-family:monospace;font-size:.7rem;color:#222244;letter-spacing:.2em;">
            MP3 · WAV · OGG · FLAC
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">SUPPORTED GENRES</div>', unsafe_allow_html=True)
    gcols = st.columns(5)
    for i, g in enumerate(GENRES):
        c = GENRE_COLORS[g]; e = GENRE_EMOJI[g]
        gcols[i%5].markdown(f"""
        <div style="background:linear-gradient(135deg,{c}0d,{c}06);
                    border:1px solid {c}22;border-radius:16px;
                    padding:18px 12px;text-align:center;margin:4px 0;">
            <div style="font-size:2rem;margin-bottom:6px;">{e}</div>
            <div style="font-family:monospace;color:{c};font-size:.78rem;
                        letter-spacing:.1em;font-weight:700;">{g.upper()}</div>
            <div style="font-family:monospace;color:#222255;font-size:.62rem;
                        letter-spacing:.1em;margin-top:2px;">{GENRE_BPM.get(g,'')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:4rem;display:grid;grid-template-columns:repeat(3,1fr);
                gap:1rem;max-width:700px;margin-left:auto;margin-right:auto;">
        <div style="background:rgba(7,7,26,.9);border:1px solid #111130;border-radius:14px;
                    padding:1.2rem;text-align:center;backdrop-filter:blur(12px);">
            <div style="font-family:monospace;color:#6444ff;font-size:1.4rem;font-weight:700;">CNN</div>
            <div style="font-family:monospace;color:#222255;font-size:.65rem;letter-spacing:.15em;margin-top:4px;">MEL SPECTROGRAMS</div>
            <div style="color:#333366;font-size:.75rem;margin-top:6px;font-family:monospace;">90.52% accuracy</div>
        </div>
        <div style="background:rgba(7,7,26,.9);border:1px solid #111130;border-radius:14px;
                    padding:1.2rem;text-align:center;backdrop-filter:blur(12px);">
            <div style="font-family:monospace;color:#ff44cc;font-size:1.4rem;font-weight:700;">LSTM</div>
            <div style="font-family:monospace;color:#222255;font-size:.65rem;letter-spacing:.15em;margin-top:4px;">MFCC SEQUENCES</div>
            <div style="color:#333366;font-size:.75rem;margin-top:6px;font-family:monospace;">81.51% accuracy</div>
        </div>
        <div style="background:rgba(7,7,26,.9);border:1px solid #111130;border-radius:14px;
                    padding:1.2rem;text-align:center;backdrop-filter:blur(12px);">
            <div style="font-family:monospace;color:#44ffaa;font-size:1.4rem;font-weight:700;">HYBRID</div>
            <div style="font-family:monospace;color:#222255;font-size:.65rem;letter-spacing:.15em;margin-top:4px;">CNN + LSTM</div>
            <div style="color:#333366;font-size:.75rem;margin-top:6px;font-family:monospace;">86.58% accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
