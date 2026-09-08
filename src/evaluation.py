"""
evaluation.py - Speech Quality Evaluation Metrics & Visualization Utilities
===========================================================================

A CSE Guide to Speech Enhancement Metrics:
------------------------------------------
1. Signal-to-Noise Ratio (SNR):
   Measures raw power ratio between ground-truth clean speech and residual noise:
     SNR = 10 * log10( ||s||^2 / ||s - s_hat||^2 )
   Limitations: Highly sensitive to overall volume scaling. If the model reduces
   volume by 10%, SNR drops drastically even if audio sounds clean.

2. Scale-Invariant Signal-to-Distortion Ratio (SI-SDR):
   The gold standard metric in Audio AI research papers (Le Roux et al., 2019).
   It first orthogonally projects s_hat onto s to find the optimal scaling factor,
   then measures the orthogonal distortion vector.
   Scale changes DO NOT penalize the score!
   - Negative SI-SDR: Severe distortion / speech drowned out.
   - 0 to +5 dB: Noticeable background noise remains.
   - +5 to +15 dB: Good to excellent noise suppression.
   - Delta SI-SDR (SI-SDR_enhanced - SI-SDR_noisy): The net improvement achieved!

3. Log-Spectral Distance (LSD):
   Measures the root-mean-square error between the log-magnitude spectrograms.
   Lower is better (0.0 dB is identical spectral envelope).
"""

from typing import Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.audio_processing import compute_stft, mag_to_db


def calculate_snr(
    clean: Union[np.ndarray, torch.Tensor],
    degraded: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-10
) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in decibels (dB).

    Parameters:
        clean: 1D array/tensor of clean reference audio
        degraded: 1D array/tensor of noisy or enhanced audio
        eps: Small epsilon to prevent division by zero

    Returns:
        snr (float): SNR in dB
    """
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    if isinstance(degraded, torch.Tensor):
        degraded = degraded.detach().cpu().numpy()

    clean = np.squeeze(clean)
    degraded = np.squeeze(degraded)

    min_len = min(len(clean), len(degraded))
    s = clean[:min_len]
    s_hat = degraded[:min_len]

    noise = s - s_hat
    signal_power = np.sum(s ** 2)
    noise_power = np.sum(noise ** 2)

    snr = 10.0 * np.log10((signal_power + eps) / (noise_power + eps))
    return float(snr)


def calculate_si_sdr(
    clean: Union[np.ndarray, torch.Tensor],
    degraded: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-10
) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Algorithm:
    1. Zero-mean center both signals.
    2. Project degraded signal onto clean signal:
       s_target = ( <s_hat, s> / ||s||^2 ) * s
    3. Calculate residual error vector:
       e_noise = s_hat - s_target
    4. SI-SDR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )
    """
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    if isinstance(degraded, torch.Tensor):
        degraded = degraded.detach().cpu().numpy()

    clean = np.squeeze(clean)
    degraded = np.squeeze(degraded)

    min_len = min(len(clean), len(degraded))
    s = clean[:min_len]
    s_hat = degraded[:min_len]

    # Zero-mean center
    s = s - np.mean(s)
    s_hat = s_hat - np.mean(s_hat)

    # Dot products
    dot_target = np.sum(s_hat * s)
    s_energy = np.sum(s ** 2) + eps

    # Orthogonal projection of s_hat onto s
    s_target = (dot_target / s_energy) * s
    e_noise = s_hat - s_target

    target_energy = np.sum(s_target ** 2) + eps
    noise_energy = np.sum(e_noise ** 2) + eps

    si_sdr = 10.0 * np.log10(target_energy / noise_energy)
    return float(si_sdr)


def calculate_lsd(
    clean_mag: np.ndarray,
    degraded_mag: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Calculate Log-Spectral Distance (LSD) in dB between two magnitude spectrograms.
    Lower values indicate closer spectral fidelity.
    """
    min_f = min(clean_mag.shape[0], degraded_mag.shape[0])
    min_t = min(clean_mag.shape[1], degraded_mag.shape[1])

    s_clean = clean_mag[:min_f, :min_t]
    s_deg = degraded_mag[:min_f, :min_t]

    log_clean = 20.0 * np.log10(np.maximum(s_clean, eps))
    log_deg = 20.0 * np.log10(np.maximum(s_deg, eps))

    # Mean squared error across frequency bins for each time frame
    diff_sq = (log_clean - log_deg) ** 2
    rmse_per_frame = np.sqrt(np.mean(diff_sq, axis=0))
    lsd = np.mean(rmse_per_frame)
    return float(lsd)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_waveform_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    title: str = "Audio Waveform Comparison (Clean vs Noisy vs Enhanced)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 3-panel time-domain waveform comparison.
    """
    min_len = min(len(clean), len(noisy), len(enhanced))
    t = np.arange(min_len) / sr

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor('#0f172a')  # Dark slate background

    panel_info = [
        (axes[0], clean[:min_len], "Original Clean Speech s(t)", "#38bdf8"),
        (axes[1], noisy[:min_len], "Corrupted Noisy Input y(t) = s(t) + n(t)", "#f87171"),
        (axes[2], enhanced[:min_len], "AI-Enhanced Output Speech s_hat(t)", "#4ade80"),
    ]

    for ax, data, subtitle, color in panel_info:
        ax.set_facecolor('#1e293b')
        ax.plot(t, data, color=color, linewidth=0.8, alpha=0.9)
        ax.set_title(subtitle, color='#f8fafc', fontsize=11, fontweight='bold', loc='left')
        ax.set_ylabel("Amplitude", color='#94a3b8', fontsize=9)
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, linestyle='--', alpha=0.2, color='#64748b')
        ax.set_ylim(-1.05, 1.05)

    axes[2].set_xlabel("Time (seconds)", color='#94a3b8', fontsize=10)
    fig.suptitle(title, color='#f8fafc', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


def plot_spectrogram_comparison(
    clean_mag: np.ndarray,
    noisy_mag: np.ndarray,
    enhanced_mag: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot high-resolution 4-panel spectral comparison:
    1. Clean Spectrogram
    2. Noisy Spectrogram
    3. Enhanced Spectrogram
    4. Predicted Filter Mask (IRM)
    """
    clean_db = mag_to_db(clean_mag)
    noisy_db = mag_to_db(noisy_mag)
    enh_db = mag_to_db(enhanced_mag)

    num_panels = 4 if mask is not None else 3
    fig, axes = plt.subplots(num_panels, 1, figsize=(12, 3.2 * num_panels), sharex=True)
    fig.patch.set_facecolor('#0f172a')

    num_freqs = clean_mag.shape[0]
    num_frames = clean_mag.shape[1]
    max_time = (num_frames * hop_length) / sr
    max_freq = (sr / 2) * (num_freqs / (n_fft // 2 + 1)) / 1000.0  # kHz

    extent = [0, max_time, 0, max_freq]

    spectrogram_specs = [
        (axes[0], clean_db, "1. Clean Target Spectrogram |S| (kHz)", "magma", "dB"),
        (axes[1], noisy_db, "2. Noisy Input Spectrogram |Y| (Voice + Noise)", "magma", "dB"),
        (axes[2], enh_db, "3. AI-Enhanced Spectrogram |S_hat| (Noise Suppressed)", "magma", "dB"),
    ]

    vmin = min(np.min(clean_db), np.min(noisy_db))
    vmax = max(np.max(clean_db), np.max(noisy_db))

    for ax, data, title_str, cmap, clabel in spectrogram_specs:
        ax.set_facecolor('#1e293b')
        im = ax.imshow(data, origin='lower', aspect='auto', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title_str, color='#f8fafc', fontsize=11, fontweight='bold', loc='left')
        ax.set_ylabel("Freq (kHz)", color='#94a3b8', fontsize=9)
        ax.tick_params(colors='#94a3b8')
        cbar = plt.colorbar(im, ax=ax, pad=0.015, aspect=20)
        cbar.set_label(clabel, color='#94a3b8', fontsize=8)
        cbar.ax.yaxis.set_tick_params(color='#94a3b8')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')

    if mask is not None:
        ax_mask = axes[3]
        ax_mask.set_facecolor('#1e293b')
        im_mask = ax_mask.imshow(mask, origin='lower', aspect='auto', extent=extent, cmap='viridis', vmin=0.0, vmax=1.0)
        ax_mask.set_title("4. AI Predicted Ideal Ratio Mask (IRM) [0.0 = Mute Noise, 1.0 = Pass Voice]",
                          color='#38bdf8', fontsize=11, fontweight='bold', loc='left')
        ax_mask.set_ylabel("Freq (kHz)", color='#94a3b8', fontsize=9)
        ax_mask.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=10)
        ax_mask.tick_params(colors='#94a3b8')
        cbar_m = plt.colorbar(im_mask, ax=ax_mask, pad=0.015, aspect=20)
        cbar_m.set_label("Gain", color='#94a3b8', fontsize=8)
        cbar_m.ax.yaxis.set_tick_params(color='#94a3b8')
        plt.setp(plt.getp(cbar_m.ax.axes, 'yticklabels'), color='#94a3b8')
    else:
        axes[-1].set_xlabel("Time (seconds)", color='#94a3b8', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig
