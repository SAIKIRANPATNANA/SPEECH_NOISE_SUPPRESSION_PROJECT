"""
generate_notebooks.py - Standalone, Google Colab-Ready Pedagogical Notebooks
=============================================================================
Generates 4 fully self-contained Jupyter notebooks with ZERO dependencies on local `src` files.
Each notebook can be directly uploaded and executed in Google Colab or JupyterLab standalone.
"""

import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

output_dir = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# COMMON DSP & UTILITY CODE STRINGS (Self-contained in notebooks)
# ==============================================================================

COMMON_COLAB_SETUP = """# 🚀 Google Colab Setup (Runs everywhere: Colab, Kaggle, Local)
# Install required audio libraries silently if not already installed
!pip install -q soundfile librosa matplotlib torch ipython

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import IPython.display as ipd
from IPython.display import display
import torch

# Clean dark-themed visualization style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
"""

COMMON_AUDIO_FUNCTIONS = """# ==============================================================================
# Standalone Audio & DSP Functions (Zero 'src' dependencies)
# ==============================================================================

def generate_synthetic_speech(duration=3.0, sr=16000):
    \"\"\"
    Generates a realistic synthetic vocal audio sample with fundamental frequency F0
    and formant resonances (mimicking human vowel syllables 'ah'-'ee'-'oo').
    Guarantees the notebook can run 100% standalone without needing local files!
    \"\"\"
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Pitch contour F0: gentle vibrato around 140 Hz (human voice pitch)
    f0 = 140.0 + 8.0 * np.sin(2 * np.pi * 3.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    
    # Glottal source: impulse-like harmonic buzz
    source = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase) + 0.2 * np.sin(4 * phase)
    
    # Formants for vocal tract vowels: F1 (700Hz), F2 (1220Hz), F3 (2600Hz)
    vocal = source * np.sin(2 * np.pi * 700 * t) + 0.5 * source * np.sin(2 * np.pi * 1220 * t) + 0.25 * source * np.sin(2 * np.pi * 2600 * t)
    
    # Syllable rhythm envelope (speech pauses and syllable bursts)
    syllables = np.maximum(0.0, np.sin(2 * np.pi * 1.8 * t)) ** 2
    speech = vocal * syllables
    
    # Normalize to [-0.9, 0.9]
    speech = speech / (np.max(np.abs(speech)) + 1e-8) * 0.9
    return speech.astype(np.float32)

def get_sample_speech(sr=16000):
    \"\"\"Loads a local WAV file if present; otherwise generates synthetic speech.\"\"\"
    search_paths = [
        "../data/clean_speech/*.wav",
        "data/clean_speech/*.wav",
        "../../data/clean_speech/*.wav"
    ]
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            wav, _ = librosa.load(matches[0], sr=sr, mono=True)
            return wav.astype(np.float32)
            
    print("ℹ️ No local WAV file found. Using high-fidelity synthetic speech sample.")
    return generate_synthetic_speech(duration=3.0, sr=sr)

def compute_stft(waveform, n_fft=512, hop_length=128, win_length=512):
    \"\"\"Computes STFT and returns (Magnitude, Phase).\"\"\"
    window = np.hanning(win_length)
    stft_matrix = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    magnitude = np.abs(stft_matrix).astype(np.float32)
    phase = np.angle(stft_matrix).astype(np.float32)
    return magnitude, phase

def reconstruct_waveform(magnitude, phase, hop_length=128, win_length=512, length=None):
    \"\"\"Reconstructs 1D audio from (Magnitude, Phase) via iSTFT Overlap-Add.\"\"\"
    complex_spec = magnitude * np.exp(1j * phase)
    window = np.hanning(win_length)
    waveform = librosa.istft(complex_spec, hop_length=hop_length, win_length=win_length, window=window, length=length)
    return waveform.astype(np.float32)

def mag_to_db(magnitude, top_db=80.0):
    \"\"\"Converts linear magnitude to decibels for visualization.\"\"\"
    mag_clamped = np.maximum(magnitude, 1e-6)
    ref = np.max(mag_clamped)
    db = 20.0 * np.log10(mag_clamped / (ref + 1e-8))
    return np.maximum(db, -top_db)
"""

# ==============================================================================
# NOTEBOOK 1: STFT, Spectrograms, and Phase
# ==============================================================================
nb1_cells = [
    md_cell("""# 🎙️ Notebook 1: Digital Audio, STFT, and the Magnitude vs Phase Mystery
### *A Computer Science Engineer's Guide to Sound Representation (Google Colab Ready)*

Welcome! If you are a Computer Science student or software engineer, audio signal processing (DSP) often feels intimidating due to heavy electrical engineering calculus.

Let's build a clear, code-first mental model:
- **Time Domain**: Audio in memory is just a 1D array of floats (`float[N]`).
- **The Core Problem**: In $y[t] = s[t] + n[t]$, speech and noise are blended into a single scalar value at every microsecond. You cannot unmix $5 = 2 + 3$ without extra dimensions!
- **STFT (Short-Time Fourier Transform)**: Our "audio decompiler". It slices the 1D audio into overlapping windows and converts it into a 2D matrix (`float[Freq, Time]`).
- **Magnitude vs Phase**:
  - **Magnitude ($|Z|$)**: "Which pitches are loud right now?" (The visual pattern human ears and brains use to recognize speech).
  - **Phase ($\angle Z$)**: "Sub-millisecond wave alignments".
- **iSTFT**: Overlap-Add (OLA) puzzle-piece reconstruction back into time-domain audio.

> ⚡ **Standalone & Colab Ready**: This notebook contains all necessary functions inline. No external project files required!
"""),

    code_cell(COMMON_COLAB_SETUP),
    code_cell(COMMON_AUDIO_FUNCTIONS),

    md_cell("""## 1. Loading and Inspecting a Raw Waveform
In code, an audio file is loaded as a 1D float array with a fixed sampling rate ($f_s = 16{,}000\\text{ Hz}$).
This means there are 16,000 floating-point numbers recorded for every 1 second of sound!
"""),

    code_cell("""# Load clean speech (uses local WAV if found, or synthetic speech on Colab)
sr = 16000
waveform = get_sample_speech(sr=sr)

print(f"Sampling Rate (sr): {sr} Hz")
print(f"Waveform Array Shape: {waveform.shape}")
print(f"Duration: {len(waveform)/sr:.2f} seconds")
print(f"Min / Max Amplitude: {waveform.min():.3f} / {waveform.max():.3f}")

# Listen to the audio directly
ipd.Audio(waveform, rate=sr)
"""),

    code_cell("""# Plot the 1D Time-Domain Waveform
time_axis = np.arange(len(waveform)) / sr

plt.figure(figsize=(12, 3), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#1e293b')
plt.plot(time_axis, waveform, color='#38bdf8', linewidth=0.8)
plt.title("1D Time-Domain Speech Waveform: s(t)", color='#f8fafc', fontsize=12, fontweight='bold')
plt.xlabel("Time (seconds)", color='#94a3b8')
plt.ylabel("Air Pressure Amplitude", color='#94a3b8')
plt.tick_params(colors='#94a3b8')
plt.grid(True, linestyle='--', alpha=0.2, color='#64748b')
plt.tight_layout()
plt.show()
"""),

    md_cell("""## 2. Computing the Short-Time Fourier Transform (STFT)
Why slice the audio into overlapping windows?
- A standard Fourier Transform over the entire file tells you *which* frequencies exist, but loses *when* they happened.
- Speech is non-stationary: you say "ba" at $t=0.2\\text{s}$ and "da" at $t=0.6\\text{s}$.
- Slicing audio with a **Hann window** of length $N_{\\text{fft}} = 512$ (32 ms) and hopping by $H = 128$ (8 ms) preserves both Frequency AND Time!
"""),

    code_cell("""# Compute STFT: returns Magnitude and Phase
magnitude, phase = compute_stft(waveform, n_fft=512, hop_length=128, win_length=512)

print(f"Magnitude Spectrogram Shape: {magnitude.shape}")
print(f"  - Rows (Frequency Bins): {magnitude.shape[0]} (from 0 Hz to {sr//2} Hz)")
print(f"  - Columns (Time Frames): {magnitude.shape[1]} (each frame = 8 ms)")
print(f"Phase Matrix Shape: {phase.shape} (values range from -pi to +pi)")
"""),

    code_cell("""# Visualize Magnitude vs Phase
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.patch.set_facecolor('#0f172a')

# 1. Magnitude (in dB)
mag_db = mag_to_db(magnitude)
im0 = axes[0].imshow(mag_db, origin='lower', aspect='auto', cmap='magma',
                    extent=[0, len(waveform)/sr, 0, sr/2000])
axes[0].set_facecolor('#1e293b')
axes[0].set_title("Magnitude Spectrogram |S| (Notice horizontal formant vowel bands!)", color='#f8fafc', fontweight='bold')
axes[0].set_ylabel("Frequency (kHz)", color='#94a3b8')
axes[0].tick_params(colors='#94a3b8')
cbar0 = plt.colorbar(im0, ax=axes[0])
cbar0.set_label("Energy (dB)", color='#94a3b8')
cbar0.ax.yaxis.set_tick_params(color='#94a3b8')
plt.setp(plt.getp(cbar0.ax.axes, 'yticklabels'), color='#94a3b8')

# 2. Phase
im1 = axes[1].imshow(phase, origin='lower', aspect='auto', cmap='twilight',
                    extent=[0, len(waveform)/sr, 0, sr/2000])
axes[1].set_facecolor('#1e293b')
axes[1].set_title("Phase Matrix (Looks like random TV static, but preserves microsecond timing!)", color='#f8fafc', fontweight='bold')
axes[1].set_xlabel("Time (seconds)", color='#94a3b8')
axes[1].set_ylabel("Frequency (kHz)", color='#94a3b8')
axes[1].tick_params(colors='#94a3b8')
cbar1 = plt.colorbar(im1, ax=axes[1])
cbar1.set_label("Radians [-pi, pi]", color='#94a3b8')
cbar1.ax.yaxis.set_tick_params(color='#94a3b8')
plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='#94a3b8')

plt.tight_layout()
plt.show()
"""),

    md_cell("""## 3. The iSTFT Roundtrip Reconstruction Test
Can we go back from the 2D matrix to the 1D audio without losing information?
Formula:
$$Z(f, t) = |Z(f, t)| \\cdot e^{j \\cdot \\angle Z(f, t)}$$
$$\\hat{x}(t) = \\text{iSTFT}(Z)$$
Let's verify mathematically:
"""),

    code_cell("""# Reconstruct 1D audio using iSTFT
reconstructed_audio = reconstruct_waveform(magnitude, phase, hop_length=128, win_length=512, length=len(waveform))

# Calculate absolute reconstruction error
recon_error = np.max(np.abs(waveform - reconstructed_audio))
print(f"Maximum Reconstruction Error: {recon_error:.2e}")
assert recon_error < 1e-4, "Reconstruction error must be virtually zero!"
print("✅ Verification Passed: iSTFT perfectly inverts STFT!")
"""),

    md_cell("""## 4. Why Noisy Phase is Reused: The Phase Swap Experiment!
What happens if we take clean speech magnitude and combine it with **completely random white noise phase**?
Let's test it and listen!
"""),

    code_cell("""# Replace real phase with completely random noise phase
random_phase = np.random.uniform(-np.pi, np.pi, size=phase.shape).astype(np.float32)
audio_random_phase = reconstruct_waveform(magnitude, random_phase, hop_length=128, win_length=512, length=len(waveform))

print("Listening test:")
print("1. Original speech with true phase:")
display(ipd.Audio(waveform, rate=sr))

print("2. Speech with clean magnitude BUT RANDOM PHASE:")
display(ipd.Audio(audio_random_phase, rate=sr))
"""),

    md_cell("""### 💡 Key Takeaway for Audio AI:
Notice that even with totally random phase, the words and syllables remain completely understandable!
The human brain extracts speech information primarily from the **Magnitude** pattern.
This is why practical speech enhancement systems (Zoom, Teams, Meeami) focus their neural networks on predicting the **clean magnitude**, while preserving the **noisy phase** for iSTFT!
""")
]

with open(os.path.join(output_dir, "01_audio_stft_and_spectrograms.ipynb"), "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# ==============================================================================
# NOTEBOOK 2: SNR and Noise Synthesis
# ==============================================================================
nb2_cells = [
    md_cell("""# 🔊 Notebook 2: Decibels, SNR, and Realistic Noise Synthesis
### *How to synthesize realistic noise and mix audio at exact SNR levels (Google Colab Ready)*

In real world communications (smartphones, Zoom, gaming headsets), microphones capture speech corrupted by environmental noise:
$$y(t) = s(t) + n(t)$$

In this notebook, we master:
1. **The Decibel (dB) Scale**: Why audio engineers use logarithmic power units.
2. **Signal-to-Noise Ratio (SNR)**: Mathematical derivation and formula.
3. **Procedural Noise Synthesis**: White noise, $1/f$ Pink noise, 60Hz Mains Hum, and Cafeteria Babble.
4. **Dynamic Mixing**: Scaling noise to achieve exact desired SNRs (e.g. -5 dB, 0 dB, +10 dB).

> ⚡ **Standalone & Colab Ready**: All DSP and noise generators are defined directly in this notebook!
"""),

    code_cell(COMMON_COLAB_SETUP),
    code_cell(COMMON_AUDIO_FUNCTIONS),

    code_cell("""# ==============================================================================
# Standalone Noise Synthesis & SNR Mixing Functions
# ==============================================================================

def calculate_power(signal):
    \"\"\"Calculates average power: P = mean(x^2).\"\"\"
    return float(np.mean(signal ** 2))

def calculate_snr(clean_signal, noisy_signal):
    \"\"\"Calculates SNR (dB): 10 * log10(P_clean / P_noise).\"\"\"
    noise = noisy_signal - clean_signal
    p_clean = calculate_power(clean_signal)
    p_noise = calculate_power(noise)
    if p_noise <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((p_clean + 1e-12) / p_noise))

def mix_at_snr(clean, noise, target_snr_db):
    \"\"\"
    Scales noise to achieve exact target SNR (dB):
    alpha = sqrt( P_clean / (P_noise * 10^(SNR/10)) )
    y = clean + alpha * noise
    \"\"\"
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)
    
    # Loop noise if shorter than clean
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean)]
    else:
        noise = noise[:len(clean)]
        
    p_clean = calculate_power(clean)
    p_noise = calculate_power(noise)
    
    if p_noise < 1e-12:
        return clean, noise, 100.0
        
    target_ratio = 10.0 ** (target_snr_db / 10.0)
    alpha = np.sqrt(p_clean / (p_noise * target_ratio))
    scaled_noise = alpha * noise
    noisy = clean + scaled_noise
    
    # Avoid clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 0.95:
        norm_factor = 0.95 / max_val
        noisy = noisy * norm_factor
        scaled_noise = scaled_noise * norm_factor
        
    actual_snr = calculate_snr(clean, noisy)
    return noisy, scaled_noise, actual_snr

def generate_white_noise(num_samples):
    \"\"\"Generates Gaussian White Noise (flat power spectrum).\"\"\"
    return np.random.randn(num_samples).astype(np.float32)

def generate_pink_noise(num_samples):
    \"\"\"Generates Pink Noise (1/f power drop: rain, waterfall, wind).\"\"\"
    white = np.random.randn(num_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = 1e-6
    fft_pink = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft_pink, n=num_samples)
    return (pink / (np.max(np.abs(pink)) + 1e-8)).astype(np.float32)

def generate_hum_noise(num_samples, sr=16000, base_freq=60.0):
    \"\"\"Generates 60 Hz electrical mains hum with odd harmonics.\"\"\"
    t = np.arange(num_samples) / sr
    hum = np.sin(2 * np.pi * base_freq * t)
    hum += 0.5 * np.sin(2 * np.pi * (base_freq * 2) * t)
    hum += 0.3 * np.sin(2 * np.pi * (base_freq * 3) * t)
    hum += 0.15 * np.sin(2 * np.pi * (base_freq * 5) * t)
    return (hum / (np.max(np.abs(hum)) + 1e-8)).astype(np.float32)

def generate_babble_noise(num_samples, sr=16000, num_speakers=4):
    \"\"\"Simulates cafeteria background babble by overlaying pitch-shifted tones.\"\"\"
    t = np.arange(num_samples) / sr
    babble = np.zeros(num_samples, dtype=np.float32)
    pitches = [110, 160, 210, 260]
    for p in pitches[:num_speakers]:
        f = p + 15.0 * np.sin(2 * np.pi * 1.2 * t)
        phase = 2 * np.pi * np.cumsum(f) / sr
        sig = np.sin(phase) * (np.maximum(0, np.sin(2 * np.pi * 3 * t)) ** 2)
        babble += sig
    return (babble / (np.max(np.abs(babble)) + 1e-8)).astype(np.float32)
"""),

    md_cell("""## 1. Demystifying Decibels (dB) and SNR
The human ear responds logarithmically to sound power:
- $+10\\text{ dB}$ means sound power is $10\\times$ higher (perceived as roughly twice as loud).
- $+20\\text{ dB}$ means $100\\times$ higher power.

The **Signal-to-Noise Ratio (SNR)** formula:
$$\\text{SNR}_{\\text{dB}} = 10 \\log_{10}\\left( \\frac{P_{\\text{speech}}}{P_{\\text{noise}}} \\right)$$
Where signal power is the mean squared amplitude: $P = \\frac{1}{N} \\sum x[t]^2$.
"""),

    code_cell("""# Load clean speech
sr = 16000
clean_speech = get_sample_speech(sr=sr)
num_samples = len(clean_speech)

p_speech = calculate_power(clean_speech)
print(f"Clean speech duration: {num_samples/sr:.2f}s | Power: {p_speech:.6f}")
"""),

    md_cell("""## 2. Generating the 4 Major Types of Noise
Let's synthesize 4 distinct noise profiles:
1. **White Noise**: Flat frequency spectrum. Radio static.
2. **Pink Noise**: $1/f$ spectrum (energy drops 3 dB per octave). Steady rain, wind.
3. **60 Hz Power Hum**: Mains hum + harmonic overtones. Ground loop, HVAC motors.
4. **Babble Noise**: Multi-speaker background voices. Coffee shop cocktail party.
"""),

    code_cell("""white_noise = generate_white_noise(num_samples)
pink_noise = generate_pink_noise(num_samples)
hum_noise = generate_hum_noise(num_samples, sr=sr, base_freq=60.0)
babble_noise = generate_babble_noise(num_samples, sr=sr, num_speakers=4)

print("Synthesized 4 noise sources successfully!")
"""),

    code_cell("""# Compare the Power Spectra of Noise Types
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
fig.patch.set_facecolor('#0f172a')

noise_dict = {
    "White Noise (Flat)": (white_noise, axes[0, 0], '#38bdf8'),
    "Pink Noise (1/f Roll-off)": (pink_noise, axes[0, 1], '#fb7185'),
    "60 Hz Mains Hum (Harmonics)": (hum_noise, axes[1, 0], '#facc15'),
    "Cafeteria Babble (Overlapping Voices)": (babble_noise, axes[1, 1], '#4ade80')
}

for title, (n_data, ax, color) in noise_dict.items():
    ax.set_facecolor('#1e293b')
    mag, _ = compute_stft(n_data)
    mean_spectrum = np.mean(mag, axis=1)
    freqs = np.linspace(0, sr / 2000, len(mean_spectrum))
    ax.plot(freqs, 20*np.log10(mean_spectrum + 1e-6), color=color, linewidth=1.2)
    ax.set_title(title, color='#f8fafc', fontsize=10, fontweight='bold')
    ax.set_xlabel("Frequency (kHz)", color='#94a3b8', fontsize=8)
    ax.set_ylabel("Power (dB)", color='#94a3b8', fontsize=8)
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, linestyle='--', alpha=0.2, color='#64748b')

plt.suptitle("Frequency Spectra of Common Audio Noise Profiles", color='#f8fafc', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""),

    md_cell("""## 3. Mixing Speech and Noise at Configurable SNR
To achieve an exact target SNR, we compute the scaling factor:
$$\\alpha = \\sqrt{\\frac{P_{\\text{speech}}}{P_{\\text{noise}} \\cdot 10^{\\text{SNR}_{\\text{target}} / 10}}}$$
$$y(t) = s(t) + \\alpha \\cdot n(t)$$
Let's create audio mixtures at $+10\\text{ dB}$, $0\\text{ dB}$, and $-5\\text{ dB}$:
"""),

    code_cell("""snr_levels = [10.0, 0.0, -5.0]
mixtures = {}

for target_snr in snr_levels:
    noisy, scaled_noise, actual_snr = mix_at_snr(clean_speech, pink_noise, target_snr_db=target_snr)
    mixtures[target_snr] = (noisy, actual_snr)
    print(f"Target SNR: {target_snr:+5.1f} dB | Measured SNR: {actual_snr:+5.2f} dB")
"""),

    code_cell("""# Audio Listening Test across SNR levels
print("1. Clean Reference Speech:")
display(ipd.Audio(clean_speech, rate=sr))

print("2. Mild Noise (+10 dB SNR) - Speech clearly audible:")
display(ipd.Audio(mixtures[10.0][0], rate=sr))

print("3. Equal Power (0 dB SNR) - Speech struggling against noise:")
display(ipd.Audio(mixtures[0.0][0], rate=sr))

print("4. Severe Noise (-5 dB SNR) - Noise 3.16x more powerful than speech:")
display(ipd.Audio(mixtures[-5.0][0], rate=sr))
""")
]

with open(os.path.join(output_dir, "02_snr_and_noise_synthesis.ipynb"), "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# ==============================================================================
# NOTEBOOK 3: Masking and U-Net Architecture
# ==============================================================================
nb3_cells = [
    md_cell("""# 🧠 Notebook 3: Masking Theory and U-Net Architecture
### *Ideal Binary Mask (IBM) vs Ideal Ratio Mask (IRM) and PyTorch U-Net from Scratch (Google Colab Ready)*

In this notebook, we bridge Digital Signal Processing and Deep Learning:
1. **The Masking Paradigm**: The dimmer switch analogy.
2. **IBM vs IRM**:
   - Why hard binary 0/1 cutoffs sound harsh and produce "musical noise".
   - Why soft $[0.0, 1.0]$ ratio masks sound smooth and natural.
3. **U-Net Architecture from Scratch in PyTorch**:
   - Convolutional encoder contracting path.
   - Bottleneck acoustic context.
   - Decoder expanding path.
   - **Skip connections**: Why they are essential to prevent muffled audio.
   - Layer-by-layer tensor shape tracing.

> ⚡ **Standalone & Colab Ready**: Contains full PyTorch model implementation inline!
"""),

    code_cell(COMMON_COLAB_SETUP),
    code_cell(COMMON_AUDIO_FUNCTIONS),

    code_cell("""# Pad or Crop Spectrogram to fixed [target_freq, target_frames]
def pad_or_crop_spectrogram(spec, target_frames=256, target_freq=256):
    freq_bins, time_frames = spec.shape
    
    # 1. Frequency axis
    if freq_bins < target_freq:
        spec = np.pad(spec, ((0, target_freq - freq_bins), (0, 0)), mode='constant')
    elif freq_bins > target_freq:
        spec = spec[:target_freq, :]
        
    # 2. Time axis
    if time_frames < target_frames:
        spec = np.pad(spec, ((0, 0), (0, target_frames - time_frames)), mode='reflect')
    elif time_frames > target_frames:
        spec = spec[:, :target_frames]
        
    return spec.astype(np.float32)

def generate_pink_noise(num_samples):
    white = np.random.randn(num_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = 1e-6
    fft_pink = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft_pink, n=num_samples)
    return (pink / (np.max(np.abs(pink)) + 1e-8)).astype(np.float32)

def mix_at_snr(clean, noise, target_snr_db=0.0):
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2) + 1e-8
    ratio = 10.0 ** (target_snr_db / 10.0)
    alpha = np.sqrt(p_clean / (p_noise * ratio))
    scaled_noise = alpha * noise
    return clean + scaled_noise, scaled_noise
"""),

    md_cell("""## 1. Deriving the Ideal Ratio Mask (IRM)
Given clean speech magnitude $|S(f, t)|$ and noise magnitude $|N(f, t)|$:
$$M_{\\text{IRM}}(f, t) = \\frac{|S(f, t)|}{|S(f, t)| + |N(f, t)| + \\epsilon} \\in [0.0, 1.0]$$

In contrast, the Ideal Binary Mask (IBM) applies a hard threshold:
$$M_{\\text{IBM}}(f, t) = \\begin{cases} 1.0 & \\text{if } |S(f, t)| > |N(f, t)| \\\\ 0.0 & \\text{otherwise} \\end{cases}$$
Let's compute and visualize both masks on actual audio!
"""),

    code_cell("""# Load clean speech and mix with pink noise at 0 dB SNR
sr = 16000
clean_audio = get_sample_speech(sr=sr)
noise = generate_pink_noise(len(clean_audio))
noisy_audio, scaled_noise = mix_at_snr(clean_audio, noise, target_snr_db=0.0)

# Compute STFTs
s_mag, _ = compute_stft(clean_audio)
n_mag, _ = compute_stft(scaled_noise)
y_mag, _ = compute_stft(noisy_audio)

# Crop to [256, 256]
s_crop = pad_or_crop_spectrogram(s_mag, 256, 256)
n_crop = pad_or_crop_spectrogram(n_mag, 256, 256)

# Compute IRM and IBM
eps = 1e-8
irm_mask = s_crop / (s_crop + n_crop + eps)
ibm_mask = (s_crop > n_crop).astype(np.float32)
"""),

    code_cell("""# Visualize IRM vs IBM
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor('#0f172a')

# Plot IBM
im0 = axes[0].imshow(ibm_mask, origin='lower', aspect='auto', cmap='bwr', vmin=0, vmax=1)
axes[0].set_facecolor('#1e293b')
axes[0].set_title("Ideal Binary Mask (IBM) - Harsh 0/1 Cutoffs (Musical Noise)", color='#f8fafc', fontweight='bold')
axes[0].set_ylabel("Frequency Bins", color='#94a3b8')
axes[0].set_xlabel("Time Frames", color='#94a3b8')
axes[0].tick_params(colors='#94a3b8')

# Plot IRM
im1 = axes[1].imshow(irm_mask, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
axes[1].set_facecolor('#1e293b')
axes[1].set_title("Ideal Ratio Mask (IRM) - Smooth Floating-Point Gains [0, 1]", color='#f8fafc', fontweight='bold')
axes[1].set_ylabel("Frequency Bins", color='#94a3b8')
axes[1].set_xlabel("Time Frames", color='#94a3b8')
axes[1].tick_params(colors='#94a3b8')
cbar = plt.colorbar(im1, ax=axes[1])
cbar.set_label("Attenuation Gain", color='#94a3b8')
cbar.ax.yaxis.set_tick_params(color='#94a3b8')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')

plt.tight_layout()
plt.show()
"""),

    md_cell("""## 2. Building the 2D U-Net Model in PyTorch from Scratch
Here is the complete U-Net implementation:
"""),

    code_cell("""import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNetSpeechEnhancer(nn.Module):
    \"\"\"
    Complete 2D Audio U-Net for Speech Noise Suppression.
    Accepts 1-channel magnitude spectrograms [Batch, 1, 256, 256].
    Outputs enhanced magnitude spectrogram and predicted mask [Batch, 1, 256, 256].
    \"\"\"
    def __init__(self, mode: str = 'masking'):
        super().__init__()
        self.mode = mode
        
        # Encoder (Contracting Path)
        self.enc1 = ConvBlock(1, 32, use_norm=False)    # -> [B, 32, 128, 128]
        self.enc2 = ConvBlock(32, 64, use_norm=True)    # -> [B, 64, 64, 64]
        self.enc3 = ConvBlock(64, 128, use_norm=True)   # -> [B, 128, 32, 32]
        self.enc4 = ConvBlock(128, 256, use_norm=True)  # -> [B, 256, 16, 16]
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), # -> [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        
        # Decoder (Expanding Path with Skip Connections)
        self.dec4 = DeconvBlock(512, 256, use_dropout=True)
        self.dec3 = DeconvBlock(512, 128, use_dropout=True)
        self.dec2 = DeconvBlock(256, 64, use_dropout=False)
        self.dec1 = DeconvBlock(128, 32, use_dropout=False)
        
        # Final Output Layer
        self.final_conv = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.output_act = nn.Sigmoid()
        
    def forward(self, noisy_mag: torch.Tensor):
        # Encoder
        e1 = self.enc1(noisy_mag)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with Skips
        d4 = self.dec4(b)
        cat4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(cat4)
        cat3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(cat3)
        cat2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(cat2)
        cat1 = torch.cat([d1, e1], dim=1)
        
        mask = self.output_act(self.final_conv(cat1))
        enhanced_mag = noisy_mag * mask
        return enhanced_mag, mask

model = UNetSpeechEnhancer()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ U-Net instantiated successfully! Trainable parameters: {total_params:,}")
"""),

    md_cell("""## 3. Tracing the Tensor Dimensions Layer-by-Layer
Let's pass a dummy mini-batch tensor `[1, 1, 256, 256]` and trace the exact shape across all stages:
"""),

    code_cell("""dummy_x = torch.randn(1, 1, 256, 256)

print(f"{'Stage':<20}{'Tensor Shape':<25}{'What is happening here?':<35}")
print("=" * 80)
print(f"{'Input':<20}{str(list(dummy_x.shape)):<25}{'1-channel noisy magnitude spectrogram'}")

# Encoder
e1 = model.enc1(dummy_x)
print(f"{'Encoder 1':<20}{str(list(e1.shape)):<25}{'Downsample 2x, 32 feature maps'}")
e2 = model.enc2(e1)
print(f"{'Encoder 2':<20}{str(list(e2.shape)):<25}{'Downsample 2x, 64 feature maps'}")
e3 = model.enc3(e2)
print(f"{'Encoder 3':<20}{str(list(e3.shape)):<25}{'Downsample 2x, 128 feature maps'}")
e4 = model.enc4(e3)
print(f"{'Encoder 4':<20}{str(list(e4.shape)):<25}{'Downsample 2x, 256 feature maps'}")

# Bottleneck
b = model.bottleneck(e4)
print(f"{'Bottleneck':<20}{str(list(b.shape)):<25}{'Deepest latent acoustic representation'}")

# Decoder with skip connections
d4 = model.dec4(b)
cat4 = torch.cat([d4, e4], dim=1)
print(f"{'Decoder 4 (+Skip4)':<20}{str(list(cat4.shape)):<25}{'Upsample 2x, Concat with Enc4'}")

d3 = model.dec3(cat4)
cat3 = torch.cat([d3, e3], dim=1)
print(f"{'Decoder 3 (+Skip3)':<20}{str(list(cat3.shape)):<25}{'Upsample 2x, Concat with Enc3'}")

d2 = model.dec2(cat3)
cat2 = torch.cat([d2, e2], dim=1)
print(f"{'Decoder 2 (+Skip2)':<20}{str(list(cat2.shape)):<25}{'Upsample 2x, Concat with Enc2'}")

d1 = model.dec1(cat2)
cat1 = torch.cat([d1, e1], dim=1)
print(f"{'Decoder 1 (+Skip1)':<20}{str(list(cat1.shape)):<25}{'Upsample 2x, Concat with Enc1'}")

# Output
enh_mag, mask = model(dummy_x)
print(f"{'Output Mask':<20}{str(list(mask.shape)):<25}{'Sigmoid filter mask in [0, 1]'}")
print(f"{'Enhanced Mag':<20}{str(list(enh_mag.shape)):<25}{'Element-wise filtered magnitude'}")
print("=" * 80)
""")
]

with open(os.path.join(output_dir, "03_masking_and_unet_architecture.ipynb"), "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

# ==============================================================================
# NOTEBOOK 4: Training, Inference, and Comprehensive Evaluation
# ==============================================================================
nb4_cells = [
    md_cell("""# 🚀 Notebook 4: Training, Audio Enhancement & Benchmarking
### *Train the U-Net, denoise corrupted audio, and compute objective metrics (Google Colab Ready)*

In this final notebook, we:
1. Load or train our PyTorch U-Net model from scratch.
2. Form dynamic training batches on-the-fly (clean speech + synthetic noise).
3. Enhance unseen noisy audio.
4. Listen to the **Before vs After** audio in real time.
5. Calculate objective audio quality metrics:
   - **$\\Delta\\text{SNR}$ (Signal-to-Noise Ratio Gain)**
   - **$\\Delta\\text{SI-SDR}$ (Scale-Invariant Signal-to-Distortion Ratio Gain)**
   - **Log-Spectral Distance (LSD)**
6. Plot the publication-ready 4-panel spectral comparison!

> ⚡ **Standalone & Colab Ready**: Trains fast on Colab GPU (~15s) or CPU (~35s). Zero external files required!
"""),

    code_cell(COMMON_COLAB_SETUP),
    code_cell(COMMON_AUDIO_FUNCTIONS),

    code_cell("""# ==============================================================================
# Standalone Model, Metrics, and Dataset
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNetSpeechEnhancer(nn.Module):
    \"\"\"
    Complete 2D Audio U-Net for Speech Noise Suppression.
    Accepts 1-channel magnitude spectrograms [Batch, 1, 256, 256].
    Outputs enhanced magnitude spectrogram and predicted mask [Batch, 1, 256, 256].
    \"\"\"
    def __init__(self, mode: str = 'masking'):
        super().__init__()
        self.mode = mode
        
        # Encoder (Contracting Path)
        self.enc1 = ConvBlock(1, 32, use_norm=False)    # -> [B, 32, 128, 128]
        self.enc2 = ConvBlock(32, 64, use_norm=True)    # -> [B, 64, 64, 64]
        self.enc3 = ConvBlock(64, 128, use_norm=True)   # -> [B, 128, 32, 32]
        self.enc4 = ConvBlock(128, 256, use_norm=True)  # -> [B, 256, 16, 16]
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), # -> [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )
        
        # Decoder (Expanding Path with Skip Connections)
        self.dec4 = DeconvBlock(512, 256, use_dropout=True)
        self.dec3 = DeconvBlock(512, 128, use_dropout=True)
        self.dec2 = DeconvBlock(256, 64, use_dropout=False)
        self.dec1 = DeconvBlock(128, 32, use_dropout=False)
        
        # Final Output Layer
        self.final_conv = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.output_act = nn.Sigmoid()
        
    def forward(self, noisy_mag: torch.Tensor):
        # Encoder
        e1 = self.enc1(noisy_mag)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with Skips
        d4 = self.dec4(b)
        cat4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(cat4)
        cat3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(cat3)
        cat2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(cat2)
        cat1 = torch.cat([d1, e1], dim=1)
        
        mask = self.output_act(self.final_conv(cat1))
        enhanced_mag = noisy_mag * mask
        return enhanced_mag, mask

def pad_or_crop_spectrogram(spec, target_frames=256, target_freq=256):
    freq_bins, time_frames = spec.shape
    if freq_bins < target_freq:
        spec = np.pad(spec, ((0, target_freq - freq_bins), (0, 0)), mode='constant')
    elif freq_bins > target_freq:
        spec = spec[:target_freq, :]
    if time_frames < target_frames:
        spec = np.pad(spec, ((0, 0), (0, target_frames - time_frames)), mode='reflect')
    elif time_frames > target_frames:
        spec = spec[:, :target_frames]
    return spec.astype(np.float32)

def calculate_snr(clean, noisy):
    noise = noisy - clean
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((p_clean + 1e-12) / p_noise))

def calculate_si_sdr(reference, estimate):
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)
    dot = np.dot(est, ref)
    ref_energy = np.dot(ref, ref) + 1e-8
    scaling = dot / ref_energy
    e_target = scaling * ref
    e_res = est - e_target
    target_p = np.sum(e_target ** 2)
    res_p = np.sum(e_res ** 2) + 1e-8
    return float(10.0 * np.log10(target_p / res_p))

def calculate_lsd(clean_mag, enh_mag):
    eps = 1e-8
    p_clean = 10.0 * np.log10(np.maximum(clean_mag, eps) ** 2)
    p_enh = 10.0 * np.log10(np.maximum(enh_mag, eps) ** 2)
    diff = (p_clean - p_enh) ** 2
    return float(np.mean(np.sqrt(np.mean(diff, axis=0))))

def generate_noise_sample(samples, sr=16000, noise_type='pink'):
    if noise_type == 'white':
        n = np.random.randn(samples)
    elif noise_type == 'hum':
        t = np.arange(samples) / sr
        n = np.sin(2 * np.pi * 60 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    else:  # pink
        w = np.random.randn(samples)
        fft = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(samples)
        freqs[0] = 1e-6
        n = np.fft.irfft(fft / np.sqrt(freqs), n=samples)
    return (n / (np.max(np.abs(n)) + 1e-8)).astype(np.float32)
"""),

    md_cell("""## 1. Fast On-The-Fly Dataset & Training
We generate on-the-fly speech-noise pairs and train our U-Net model.
If a trained checkpoint is found locally (`../models/best_model.pt`), we load it.
Otherwise, we run 5 quick training epochs right here!
"""),

    code_cell("""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

class StandaloneAudioDataset(Dataset):
    \"\"\"Generates paired clean speech and noisy mixtures on the fly.\"\"\"
    def __init__(self, num_samples=64, sr=16000):
        self.num_samples = num_samples
        self.sr = sr
        # Pre-generate base clean speeches
        self.clean_bank = [
            generate_synthetic_speech(duration=2.5, sr=sr) for _ in range(8)
        ]
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        clean = self.clean_bank[idx % len(self.clean_bank)]
        noise = generate_noise_sample(len(clean), sr=self.sr, noise_type='pink')
        
        # Random SNR in [-5, 10] dB
        snr_db = np.random.uniform(-5.0, 10.0)
        p_c = np.mean(clean ** 2)
        p_n = np.mean(noise ** 2) + 1e-8
        alpha = np.sqrt(p_c / (p_n * (10 ** (snr_db / 10))))
        scaled_noise = alpha * noise
        noisy = clean + scaled_noise
        
        s_mag, _ = compute_stft(clean)
        n_mag, _ = compute_stft(scaled_noise)
        y_mag, _ = compute_stft(noisy)
        
        s_crop = pad_or_crop_spectrogram(s_mag, 256, 256)
        n_crop = pad_or_crop_spectrogram(n_mag, 256, 256)
        y_crop = pad_or_crop_spectrogram(y_mag, 256, 256)
        
        irm = s_crop / (s_crop + n_crop + 1e-8)
        
        return (
            torch.from_numpy(y_crop).unsqueeze(0),
            torch.from_numpy(s_crop).unsqueeze(0),
            torch.from_numpy(irm).unsqueeze(0)
        )

# Initialize model
model = UNetSpeechEnhancer().to(device)

# Check if local trained checkpoint exists
ckpt_path = "../models/best_model.pt"
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded pre-trained checkpoint from {ckpt_path}!")
else:
    print("Training 5 fast epochs in-memory...")
    train_loader = DataLoader(StandaloneAudioDataset(num_samples=48), batch_size=8, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    crit_l1 = nn.L1Loss()
    crit_mse = nn.MSELoss()
    
    model.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        for y_in, s_target, irm_target in train_loader:
            y_in = y_in.to(device)
            s_target = s_target.to(device)
            irm_target = irm_target.to(device)
            
            optimizer.zero_grad()
            enh_mag, mask = model(y_in)
            loss = crit_l1(enh_mag, s_target) + 0.5 * crit_mse(mask, irm_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}/5 | Loss: {total_loss/len(train_loader):.4f}")
    print("✅ Training completed!")
"""),

    md_cell("""## 2. Creating an Unseen Corrupted Test Audio File
Let's create a realistic test audio clip corrupted with pink noise at $0\\text{ dB}$ SNR (speech and noise have equal power):
"""),

    code_cell("""sr = 16000
clean_test = get_sample_speech(sr=sr)
noise_test = generate_noise_sample(len(clean_test), sr=sr, noise_type='pink')

# Mix at 0 dB SNR
p_c = np.mean(clean_test ** 2)
p_n = np.mean(noise_test ** 2) + 1e-8
scaled_noise = np.sqrt(p_c / p_n) * noise_test
noisy_test = clean_test + scaled_noise

actual_snr = calculate_snr(clean_test, noisy_test)
print(f"Created Test Audio at {actual_snr:.2f} dB SNR.")
"""),

    md_cell("""## 3. Running U-Net Audio Enhancement
We pass the noisy audio through the U-Net:
"""),

    code_cell("""# 1. Compute STFT of noisy audio
noisy_mag, noisy_phase = compute_stft(noisy_test)
orig_frames = noisy_mag.shape[1]

# 2. Pad to [256, 256] for U-Net
noisy_crop = pad_or_crop_spectrogram(noisy_mag, 256, 256)
in_tensor = torch.from_numpy(noisy_crop).unsqueeze(0).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    enh_crop, mask_crop = model(in_tensor)
    enh_crop_np = enh_crop.squeeze().cpu().numpy()
    mask_crop_np = mask_crop.squeeze().cpu().numpy()

# 3. Restore original frequency and time dimensions
enh_full = np.zeros_like(noisy_mag)
min_freq = min(noisy_mag.shape[0], enh_crop_np.shape[0])
min_time = min(noisy_mag.shape[1], enh_crop_np.shape[1])
enh_full[:min_freq, :min_time] = enh_crop_np[:min_freq, :min_time]

# 4. Reconstruct 1D audio via iSTFT with original noisy phase
enhanced_test = reconstruct_waveform(enh_full, noisy_phase, length=len(noisy_test))
clean_mag, _ = compute_stft(clean_test)

print("✅ Enhancement complete!")
"""),

    md_cell("""## 4. Listening to Before vs After Audio
Listen to the clean reference, the noisy input, and the AI-enhanced output:
"""),

    code_cell("""print("1. Clean Reference Speech:")
display(ipd.Audio(clean_test, rate=sr))

print("2. Corrupted Noisy Input (0 dB SNR):")
display(ipd.Audio(noisy_test, rate=sr))

print("3. AI-Enhanced Speech (Noise Stripped):")
display(ipd.Audio(enhanced_test, rate=sr))
"""),

    md_cell("""## 5. Quantitative Audio Quality Benchmarks
Let's compute the objective metrics:
"""),

    code_cell("""snr_in = calculate_snr(clean_test, noisy_test)
snr_out = calculate_snr(clean_test, enhanced_test)
delta_snr = snr_out - snr_in

sdr_in = calculate_si_sdr(clean_test, noisy_test)
sdr_out = calculate_si_sdr(clean_test, enhanced_test)
delta_sdr = sdr_out - sdr_in

lsd_in = calculate_lsd(clean_mag, noisy_mag)
lsd_out = calculate_lsd(clean_mag, enh_full)

print("=" * 60)
print(f"{'Metric':<25}{'Noisy Input':<14}{'AI Enhanced':<14}{'Gain':<12}")
print("=" * 60)
print(f"{'SNR (dB)':<25}{snr_in:<14.2f}{snr_out:<14.2f}{delta_snr:+<12.2f}")
print(f"{'SI-SDR (dB)':<25}{sdr_in:<14.2f}{sdr_out:<14.2f}{delta_sdr:+<12.2f}")
print(f"{'Log-Spectral Dist (LSD)':<25}{lsd_in:<14.2f}{lsd_out:<14.2f}{lsd_out - lsd_in:+<12.2f}")
print("=" * 60)
"""),

    md_cell("""## 6. Visualizing the 4-Panel Spectrogram Comparison
"""),

    code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.patch.set_facecolor('#0f172a')

panels = [
    ("1. Clean Reference Speech", clean_mag, 'magma', axes[0, 0], True),
    ("2. Noisy Corrupted Input (0 dB)", noisy_mag, 'magma', axes[0, 1], True),
    ("3. AI-Enhanced Clean Speech", enh_full, 'magma', axes[1, 0], True),
    ("4. Predicted U-Net Mask (Gain)", mask_crop_np, 'viridis', axes[1, 1], False)
]

for title, data, cmap, ax, is_db in panels:
    ax.set_facecolor('#1e293b')
    if is_db:
        plot_data = mag_to_db(data)
        vmin, vmax = -60, 0
    else:
        plot_data = data
        vmin, vmax = 0.0, 1.0
        
    im = ax.imshow(plot_data, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')
    ax.set_title(title, color='#f8fafc', fontsize=11, fontweight='bold')
    ax.set_xlabel("Time Frames", color='#94a3b8', fontsize=9)
    ax.set_ylabel("Frequency Bins", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8')
    ax.grid(False)

plt.tight_layout()
plt.show()
"""),

    md_cell("""### 🎉 Conclusion
You have built, trained, and evaluated an end-to-end Deep Learning Speech Noise Suppression system in Google Colab!
Key concepts mastered:
- 1D audio $\\to$ 2D STFT decomposition.
- Why soft masking (IRM) avoids musical noise.
- How U-Net skip connections preserve crisp speech harmonics.
- How to quantitatively evaluate audio AI models using $\\Delta\\text{SNR}$ and SI-SDR.
""")
]

with open(os.path.join(output_dir, "04_train_and_evaluate_enhancement.ipynb"), "w") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

print("Successfully regenerated all 4 notebooks as 100% standalone, Google Colab-ready notebooks!")
