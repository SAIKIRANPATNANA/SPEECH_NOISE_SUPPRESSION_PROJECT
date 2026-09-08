"""
app.py - Vibrant Interactive Streamlit Web Application
======================================================
AI-Based Speech Noise Suppression with PyTorch U-Net & DSP Pipeline.

Designed specifically with intuitive Computer Science explanations,
real-time audio mixing, spectrogram visualization, and before/after comparisons.
"""

import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import glob
import tempfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st
import torch

from src.audio_processing import (
    load_audio,
    save_audio,
    compute_stft,
    mag_to_db,
    reconstruct_waveform
)
from src.noise_generator import (
    mix_at_snr,
    generate_white_noise,
    generate_pink_noise,
    generate_hum_noise,
    generate_babble_noise
)
from src.model import UNetSpeechEnhancer
from src.inference import load_model, enhance_audio
from src.evaluation import calculate_snr, calculate_si_sdr, calculate_lsd

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Speech Noise Suppressor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Vibrant CSS Design
st.markdown("""
<style>
    /* Main container and theme */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Header card */
    .hero-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    /* Metric pill boxes */
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-val.green {
        color: #34d399;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* Audio section cards */
    .audio-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    /* Sidebar aesthetic */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_model(checkpoint_path: str):
    """Load model once into memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(checkpoint_path):
        return load_model(checkpoint_path, device=device), device
    return None, device


def plot_interactive_spectrograms(clean_mag, noisy_mag, enh_mag, mask, sr=16000):
    """Render 4-panel dark theme comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    fig.patch.set_facecolor('#0f172a')
    
    panels = [
        ("Clean Reference Speech", clean_mag, 'magma', axes[0, 0], True),
        ("Noisy Corrupted Speech", noisy_mag, 'magma', axes[0, 1], True),
        ("AI-Enhanced Speech", enh_mag, 'magma', axes[1, 0], True),
        ("Predicted U-Net Mask (Gain)", mask, 'viridis', axes[1, 1], False)
    ]
    
    for title, data, cmap, ax, is_db in panels:
        ax.set_facecolor('#1e293b')
        if is_db and data is not None:
            plot_data = mag_to_db(data)
            vmin, vmax = -60, 0
        else:
            plot_data = data
            vmin, vmax = 0.0, 1.0
            
        if plot_data is not None:
            im = ax.imshow(plot_data, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='#94a3b8')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')
        else:
            ax.text(0.5, 0.5, "Not Available", color='#94a3b8', ha='center', va='center')
            
        ax.set_title(title, color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Time Frames (8ms steps)", color='#94a3b8', fontsize=9)
        ax.set_ylabel("Frequency Bins (0 - 8kHz)", color='#94a3b8', fontsize=9)
        ax.tick_params(colors='#94a3b8')
        ax.grid(False)

    plt.tight_layout()
    return fig


def main():
    # Header Banner
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🎙️ AI-Based Speech Noise Suppression Studio</div>
        <div class="hero-subtitle">
            An end-to-end deep learning speech enhancement platform built from scratch in PyTorch.<br>
            <strong>Pipeline:</strong> Clean speech + Noise &rarr; <strong>STFT</strong> &rarr; 
            <strong>U-Net Mask Prediction</strong> &rarr; <strong>iSTFT</strong> &rarr; Enhanced Intelligible Speech.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Checkpoint and Device Setup
    default_ckpt = "SpeechNoiseSuppression/models/best_model.pt"
    if not os.path.exists(default_ckpt):
        default_ckpt = "models/best_model.pt"
        
    model, device = get_cached_model(default_ckpt)
    
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Experiment Controls")
        
        # Audio Source Selection
        audio_choice = st.radio(
            "1. Select Clean Speech Source",
            ["Preset LibriSpeech Samples", "Upload Your Own Audio"]
        )
        
        clean_audio = None
        sr = 16000
        
        clean_dir = "SpeechNoiseSuppression/data/clean_speech"
        if not os.path.exists(clean_dir):
            clean_dir = "data/clean_speech"
            
        sample_files = sorted(glob.glob(f"{clean_dir}/*.wav"))
        
        if audio_choice == "Preset LibriSpeech Samples":
            if sample_files:
                selected_sample = st.selectbox(
                    "Choose Speech Sample",
                    sample_files,
                    format_func=lambda x: os.path.basename(x)
                )
                clean_audio, sr = load_audio(selected_sample, target_sr=16000)
            else:
                st.warning("No clean speech samples found in data/clean_speech.")
        else:
            uploaded_file = st.file_uploader("Upload Clean Audio (WAV/MP3)", type=["wav", "mp3", "flac"])
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                clean_audio, sr = load_audio(tmp_path, target_sr=16000)
                os.remove(tmp_path)
                
        st.markdown("---")
        
        # Noise Selection
        st.subheader("2. Add Background Noise")
        noise_type = st.selectbox(
            "Noise Type",
            ["Pink Noise (Rain/Wind)", "60Hz Mains Hum (HVAC/Ground)", "White Noise (Radio Static)", "Cafeteria Babble (Chatter)"]
        )
        
        # SNR Slider
        snr_target = st.slider(
            "Signal-to-Noise Ratio (SNR in dB)",
            min_value=-10.0,
            max_value=15.0,
            value=0.0,
            step=1.0,
            help="Lower SNR = louder noise. 0 dB means speech and noise have equal power."
        )
        
        st.markdown("---")
        st.markdown(f"**Compute Device:** `{device}`")
        if model is not None:
            st.success(f"Loaded U-Net ({model.get_num_parameters():,} params)")
        else:
            st.warning("Model checkpoint not yet created. Train model via `src/train.py`.")
            
    # Main Application Interface
    if clean_audio is None:
        st.info("👈 Please select or upload a speech audio sample from the sidebar to begin!")
        return
        
    # Generate Noise
    num_samples = len(clean_audio)
    if noise_type == "White Noise (Radio Static)":
        noise_raw = generate_white_noise(num_samples)
    elif noise_type == "Pink Noise (Rain/Wind)":
        noise_raw = generate_pink_noise(num_samples)
    elif noise_type == "60Hz Mains Hum (HVAC/Ground)":
        noise_raw = generate_hum_noise(num_samples, sr=sr, base_freq=60.0)
    else:  # Babble
        noise_raw = generate_babble_noise(sample_files, num_samples, sr=sr, num_speakers=4)
        
    # Mix at selected SNR
    noisy_audio, scaled_noise, actual_snr = mix_at_snr(clean_audio, noise_raw, target_snr_db=snr_target)
    
    # Run U-Net Enhancement
    if model is not None:
        enhanced_audio, noisy_mag, enh_mag, mask = enhance_audio(model, noisy_audio, sr=sr, device=device)
        clean_mag, _ = compute_stft(clean_audio)
        
        # Calculate Metrics
        snr_in = calculate_snr(clean_audio, noisy_audio)
        snr_out = calculate_snr(clean_audio, enhanced_audio)
        delta_snr = snr_out - snr_in
        
        sdr_in = calculate_si_sdr(clean_audio, noisy_audio)
        sdr_out = calculate_si_sdr(clean_audio, enhanced_audio)
        delta_sdr = sdr_out - sdr_in
        
        lsd_in = calculate_lsd(clean_mag, noisy_mag)
        lsd_out = calculate_lsd(clean_mag, enh_mag)
        delta_lsd = lsd_out - lsd_in
    else:
        enhanced_audio = noisy_audio
        noisy_mag, enh_mag, mask, clean_mag = None, None, None, None
        snr_in, snr_out, delta_snr = actual_snr, actual_snr, 0.0
        sdr_in, sdr_out, delta_sdr = 0.0, 0.0, 0.0
        lsd_in, lsd_out, delta_lsd = 0.0, 0.0, 0.0

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{snr_in:.1f} dB</div>
            <div class="metric-lbl">Input SNR</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val green">{snr_out:.1f} dB</div>
            <div class="metric-lbl">Enhanced SNR</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val green">+{delta_snr:.1f} dB</div>
            <div class="metric-lbl">SNR Improvement (&Delta;SNR)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val green">+{delta_sdr:.1f} dB</div>
            <div class="metric-lbl">SI-SDR Improvement</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Audio Playback Section
    st.subheader("🎧 Audio Comparison (Before vs After)")
    a_col1, a_col2, a_col3 = st.columns(3)
    
    with a_col1:
        st.markdown("**1. Original Clean Speech** (Ground Truth)")
        st.audio(clean_audio, sample_rate=sr)
        
    with a_col2:
        st.markdown(f"**2. Corrupted Audio** ({actual_snr:.1f} dB SNR)")
        st.audio(noisy_audio, sample_rate=sr)
        
    with a_col3:
        st.markdown(f"**3. AI Enhanced Speech** (+{delta_snr:.1f} dB Gain)")
        st.audio(enhanced_audio, sample_rate=sr)

    st.markdown("---")

    # Visualizations
    st.subheader("📊 Spectral Analysis: How the AI Removes Noise")
    if clean_mag is not None:
        fig = plot_interactive_spectrograms(clean_mag, noisy_mag, enh_mag, mask, sr=sr)
        st.pyplot(fig)
        
    # CSE Deep Dive Explanations Expander
    with st.expander("💡 CSE Architecture Deep Dive: How does this work?"):
        st.markdown("""
        ### Why STFT and Masking?
        1. **In Time Domain ($1D$)**: If an audio sample is `0.7`, and noise is `0.4`, speech was `0.3`. The computer cannot know which part was speech from a single number ($0.7$).
        2. **In Frequency Domain ($2D$)**: Human speech forms distinct horizontal harmonic bands (formants) between $200\\text{ Hz}$ and $4000\\text{ Hz}$. White/Pink noise is smeared across all frequencies.
        3. **The U-Net Mask**: Instead of generating speech raw, the U-Net predicts a **floating-point attenuation mask** $M(f, t) \in [0.0, 1.0]$.
           - $M = 1.0$: Keep this frequency bin untouched (speech harmonic).
           - $M = 0.0$: Mute this frequency bin completely (pure noise).
        4. **iSTFT Overlap-Add**: We multiply $|Y(f, t)| \cdot M(f, t)$, attach the noisy phase $\\angle Y$, and invert back to crisp $16\\text{ kHz}$ PCM audio!
        """)


if __name__ == "__main__":
    main()
