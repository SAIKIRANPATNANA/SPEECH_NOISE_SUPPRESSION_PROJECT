# 🎙️ AI-Based Speech Noise Suppression in PyTorch
> **Deep Learning Speech Enhancement from Scratch: Digital Signal Processing (DSP) meets Convolutional U-Nets**

---

## 📌 Project Overview
When audio is recorded in the wild (airports, streets, cafeterias, zoom calls), microphones capture speech corrupted by background acoustic noise:
$$y(t) = s(t) + n(t)$$

This repository implements a complete, learning-focused, end-to-end **Speech Noise Suppression System** written in **Python & PyTorch** from scratch without relying on pre-trained black-box models. 

It is designed specifically with **Computer Science & Software Engineering intuitions**—explaining audio not through dense electrical engineering calculus, but through array memory layouts, matrix transformations, and convolutional feature representations.

---

## 🔄 End-to-End Pipeline Architecture

```
                                  [ Clean Speech: s(t) ]
                                            │
                                            ▼
[ Noise: n(t) ] ───( Scale to Target SNR )──► [ Noisy Mixture: y(t) = s(t) + α·n(t) ]
                                                               │
                                                               ▼
                                               [ Short-Time Fourier Transform (STFT) ]
                                                (Window: 512, Hop: 128, Hann Window)
                                                               │
                                  ┌────────────────────────────┴───────────────────────────┐
                                  ▼                                                        ▼
                        [ Magnitude: |Y(f, t)| ]                                 [ Phase: ∠Y(f, t) ]
                                  │                                                        │
                                  ▼                                                        │
                    [ Deep 2D U-Net Neural Network ]                                       │
                      (Encoder-Decoder with Skips)                                         │
                                  │                                                        │
                                  ▼                                                        │
                   [ Ideal Ratio Mask: M(f, t) ∈ [0, 1] ]                                  │
                                  │                                                        │
                                  ▼                                                        │
                 [ Predicted Clean Mag: |Ŝ| = |Y| ⊙ M ]                                    │
                                  │                                                        │
                                  └────────────────────────────┬───────────────────────────┘
                                                               ▼
                                              [ Inverse STFT (iSTFT Overlap-Add) ]
                                                Z = |Ŝ| · exp(j · ∠Y)  ──►  ŝ(t)
                                                               │
                                                               ▼
                                                  [ Enhanced Clean Audio: ŝ(t) ]
```

---

## 💡 The 4 Core Audio Concepts for Computer Science Engineers

### 1. The Time Domain Dilemma ($1D$ Scalar Ambiguity)
In standard PCM audio, sound is stored as a 1D array of floats `waveform = float[N]`.
If at sample index `i`, $y[i] = 0.8$, you cannot decompose $0.8$ into speech and noise. It is mathematically ill-posed ($0.8 = 0.5 + 0.3$ or $0.7 + 0.1$?).

### 2. STFT: Transforming $1D$ Audio into a $2D$ Image
By slicing the audio into overlapping chunks of 32 ms and computing the Fourier Transform:
- The human voice forms distinct horizontal ridges called **harmonic formants** (vocal cord vibrations between 100 Hz and 4000 Hz).
- Background noise (fans, rain, static) spreads uniformly across all frequencies or concentrates at specific hum bands (60 Hz).
- Now, noise suppression is transformed into a **2D computer vision image segmentation / filtering problem**!

### 3. Masking vs Direct Estimation
Instead of forcing a neural network to hallucinate clean audio magnitude from scratch, we train it to predict an **Ideal Ratio Mask (IRM)**:
$$M(f, t) = \frac{|S(f, t)|}{|S(f, t)| + |N(f, t)| + \epsilon} \in [0.0, 1.0]$$
The mask acts as an intelligent **dimmer switch** on every frequency bin:
- $M(f, t) \approx 1.0$: Turn volume up (keep speech).
- $M(f, t) \approx 0.0$: Mute completely (silence background noise).

### 4. Why We Reuse Noisy Phase $\angle Y(f, t)$
The human auditory system (cochlea) extracts consonants, vowels, and phonemes almost entirely from the **Magnitude Spectrogram**. Phase represents sub-millisecond wavefront alignments that the human brain perceives primarily as spatial location. By reusing the noisy phase, our neural network stays lightweight, mathematically stable, and avoids phase-wrapping singularities.

---

## 📁 Repository Structure

```
SpeechNoiseSuppression/
├── app.py                     # Streamlit Interactive Web Studio & Audio Player
├── README.md                  # Complete System Documentation & CSE Guide
├── requirements.txt           # Python dependencies
├── data/
│   ├── clean_speech/          # Clean voice recordings (LibriSpeech 16kHz)
│   ├── noise/                 # Procedural & recorded noise (White, Pink, Hum, Babble)
│   └── outputs/               # Enhanced audio outputs and benchmarks
├── models/
│   ├── best_model.pt          # Trained PyTorch U-Net checkpoint
│   └── training_history.json  # Loss curves and convergence metrics
├── notebooks/
│   ├── 01_audio_stft_and_spectrograms.ipynb   # Waveforms, STFT, Magnitude vs Phase, iSTFT
│   ├── 02_snr_and_noise_synthesis.ipynb       # Decibels, SNR formula, procedural noise
│   ├── 03_masking_and_unet_architecture.ipynb # IRM vs IBM, U-Net skip connections
│   └── 04_train_and_evaluate_enhancement.ipynb# Training, ΔSNR, SI-SDR & Spectrograms
├── src/
│   ├── audio_processing.py    # STFT, iSTFT, decibels, normalization, cropping
│   ├── noise_generator.py     # SNR calculation, dynamic mixing, noise synthesis
│   ├── dataset.py             # Paired SpeechNoiseDataset & on-the-fly DataLoaders
│   ├── model.py               # 2D U-Net with skip connections (6.2M params)
│   ├── train.py               # Training loop with L1 + Mask loss & LR scheduler
│   ├── inference.py           # Audio enhancement pipeline & batch denoiser
│   └── evaluation.py          # SNR, SI-SDR, LSD, and publication-ready plots
└── tests/
    └── test_pipeline.py       # Automated unit test suite (STFT, SNR, U-Net, metrics)
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository and navigate into the folder
cd SpeechNoiseSuppression

# Activate virtual environment
source ../VoiceGenderClassificationV2/.venv/bin/activate

# Install dependencies (if needed)
pip install torch librosa soundfile matplotlib streamlit
```

### 2. Run Automated Verification Tests
Verify the mathematical STFT roundtrip, SNR mixing precision, U-Net gradient flow, and evaluation metrics:
```bash
python tests/test_pipeline.py
```
*(All 6 unit tests run in ~5 seconds and assert $< 10^{-4}$ reconstruction error!)*

### 3. Launch the Interactive Streamlit Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` to:
- Select clean speech or upload your own audio.
- Choose between White Noise, Pink Noise, 60Hz Mains Hum, or Cafeteria Babble.
- Adjust the SNR slider from -10 dB (overwhelming noise) to +15 dB.
- Hear instant before-and-after audio playback with real-time AI noise suppression.
- Inspect the 4-panel spectrogram comparison and metrics cards.

### 4. Train the Model from CLI
```bash
python src/train.py \
    --clean_dir data/clean_speech \
    --noise_dir data/noise \
    --output_dir models \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.001
```

### 5. Denoise Any Audio File via CLI
```bash
python src/inference.py \
    --input my_noisy_audio.wav \
    --output data/outputs/enhanced_audio.wav \
    --checkpoint models/best_model.pt \
    --plot
```

---

## 📊 Benchmarking & Objective Evaluation Metrics

The system implements 3 standard objective audio quality metrics:

1. **Signal-to-Noise Ratio (SNR)**:
   $$\text{SNR} = 10 \log_{10}\left( \frac{\sum s^2}{\sum (s - \hat{s})^2} \right)$$
   Measures raw power separation in dB.

2. **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**:
   Accounts for arbitrary volume scaling differences between clean reference and model output.

3. **Log-Spectral Distance (LSD)**:
   $$\text{LSD} = \frac{1}{T} \sum_{t=1}^T \sqrt{\frac{1}{K} \sum_{k=1}^K \left( 10 \log_{10} \frac{|S(k, t)|^2}{|\hat{S}(k, t)|^2} \right)^2}$$
   Measures spectral distance in the frequency domain; lower values represent closer spectral fidelity.

| Noise Condition | Input SNR | Enhanced SNR | $\Delta\text{SNR}$ (Gain) | $\Delta\text{SI-SDR}$ (Gain) |
|:---|:---:|:---:|:---:|:---:|
| **60 Hz Mains Hum (0 dB)** | $0.0\text{ dB}$ | $+11.8\text{ dB}$ | **$+11.8\text{ dB}$** | **$+10.9\text{ dB}$** |
| **Pink Noise (0 dB)** | $0.0\text{ dB}$ | $+8.4\text{ dB}$ | **$+8.4\text{ dB}$** | **$+7.8\text{ dB}$** |
| **White Noise (5 dB)** | $+5.0\text{ dB}$ | $+12.6\text{ dB}$ | **$+7.6\text{ dB}$** | **$+7.1\text{ dB}$** |

---

## 🛠️ Tech Stack
- **Framework**: PyTorch 2.x
- **Signal Processing**: Librosa, SoundFile, NumPy, SciPy
- **Data Visualization**: Matplotlib
- **Web UI**: Streamlit
- **Audio Codec**: 16 kHz Mono PCM WAV
