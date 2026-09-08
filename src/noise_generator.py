"""
noise_generator.py - Realistic Noise Synthesis and SNR Mixing
=============================================================

A CSE Guide to Noise in Audio AI:
---------------------------------
1. What is Signal-to-Noise Ratio (SNR)?
   SNR measures the ratio between the power of meaningful speech vs unwanted noise.
   Formula: SNR_dB = 10 * log10( P_speech / P_noise )
   Where Power P = (1 / N) * sum( x[t]^2 ) = RMS^2.

   Decibel (dB) intuition for software engineers:
     +20 dB -> Speech is 100x more powerful than noise (barely noticeable whisper of noise)
     +10 dB -> Speech is 10x more powerful than noise (typical office / living room)
       0 dB -> Speech and noise have EQUAL power (difficult to hear speech clearly)
      -5 dB -> Noise is ~3.16x more powerful than speech (car engine roaring / loud club)
     -10 dB -> Noise is 10x more powerful than speech (speech buried under severe distortion)

2. Types of Noise in the Real World:
   - White Noise: Equal energy across all frequencies (flat spectrum). Sounds like static / radio hiss.
   - Pink Noise: Energy rolls off as 1/f (3 dB drop per octave). Sounds like steady waterfall / wind.
   - Power Hum: Strong tonal line at 50 Hz (Europe/Asia) or 60 Hz (USA) + harmonics. AC mains / HVAC.
   - Babble Noise: Overlapping voices in a cafe / restaurant (Cocktail Party problem). Non-stationary!
"""

import os
import glob
from typing import Tuple, List, Optional, Union
import numpy as np
import soundfile as sf
import librosa


def calculate_power(signal: np.ndarray) -> float:
    """Calculate the average power (mean squared amplitude) of a 1D signal."""
    return float(np.mean(signal ** 2)) + 1e-10


def mix_at_snr(
    clean_speech: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Mix clean speech and noise at an exact target Signal-to-Noise Ratio (SNR).

    Algorithm:
    1. Align lengths: if noise is shorter, tile it; if longer, randomly slice.
    2. Compute speech power: P_speech = mean(clean^2)
    3. Compute noise power: P_noise = mean(noise^2)
    4. Calculate required noise scale factor:
       target_snr = 10 * log10(P_speech / (alpha^2 * P_noise))
       => alpha = sqrt( P_speech / (P_noise * 10^(target_snr / 10)) )
    5. Noisy signal = clean + alpha * noise

    Parameters:
        clean_speech (np.ndarray): 1D array of clean speech samples
        noise (np.ndarray): 1D array of noise samples
        target_snr_db (float): Desired SNR in decibels (e.g., -5.0, 0.0, 5.0, 10.0)

    Returns:
        noisy_signal (np.ndarray): The mixed audio
        scaled_noise (np.ndarray): The scaled noise component
        actual_snr_db (float): The verified SNR achieved
    """
    clean_len = len(clean_speech)
    noise_len = len(noise)

    # 1. Match noise length to clean speech
    if noise_len < clean_len:
        repeats = int(np.ceil(clean_len / noise_len))
        aligned_noise = np.tile(noise, repeats)[:clean_len]
    elif noise_len > clean_len:
        # Pick a random slice to introduce diversity during training
        start_idx = np.random.randint(0, noise_len - clean_len + 1)
        aligned_noise = noise[start_idx:start_idx + clean_len]
    else:
        aligned_noise = noise.copy()

    # 2. Compute powers
    p_speech = calculate_power(clean_speech)
    p_noise = calculate_power(aligned_noise)

    # 3. Calculate scaling factor
    # 10^(SNR/10) = P_speech / P_scaled_noise
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    alpha = np.sqrt(p_speech / (p_noise * snr_linear + 1e-10))

    scaled_noise = alpha * aligned_noise
    noisy_signal = clean_speech + scaled_noise

    # Verify actual SNR
    actual_p_noise = calculate_power(scaled_noise)
    actual_snr = 10.0 * np.log10(p_speech / (actual_p_noise + 1e-10))

    # Soft normalization safeguard to avoid clipping if sum exceeds [-1, 1]
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 1.0:
        noisy_signal = noisy_signal / max_val
        scaled_noise = scaled_noise / max_val

    return (
        noisy_signal.astype(np.float32),
        scaled_noise.astype(np.float32),
        float(actual_snr)
    )


# ---------------------------------------------------------------------------
# Synthetic Noise Generators (Built-in offline support)
# ---------------------------------------------------------------------------

def generate_white_noise(num_samples: int) -> np.ndarray:
    """
    Generate Gaussian white noise (flat power spectrum across all frequencies).
    """
    noise = np.random.normal(0.0, 0.1, size=num_samples)
    return (noise / np.max(np.abs(noise))).astype(np.float32)


def generate_pink_noise(num_samples: int) -> np.ndarray:
    """
    Generate Pink Noise (1/f power spectrum: equal energy per octave).
    Simulates natural background noise like heavy wind or ocean surf.
    Uses the Voss-McCartney algorithm via frequency filtering.
    """
    white = np.random.randn(num_samples)
    # FFT into frequency domain
    white_fft = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(num_samples)
    frequencies[0] = 1e-6  # Avoid division by zero at DC
    # Scale amplitudes by 1 / sqrt(f) to achieve 1/f power spectrum
    pink_fft = white_fft / np.sqrt(frequencies)
    pink = np.fft.irfft(pink_fft, n=num_samples)
    return (pink / np.max(np.abs(pink))).astype(np.float32)


def generate_hum_noise(
    num_samples: int,
    sr: int = 16000,
    base_freq: float = 60.0,
    harmonics: Optional[List[float]] = None
) -> np.ndarray:
    """
    Generate electrical mains / HVAC hum noise with harmonic overtones.
    Default: 60 Hz fundamental (US standard) + 120 Hz, 180 Hz, 240 Hz harmonics.
    """
    if harmonics is None:
        harmonics = [base_freq * 2, base_freq * 3, base_freq * 4]

    t = np.linspace(0, num_samples / sr, num_samples, endpoint=False)
    
    # Fundamental tone
    signal = 0.5 * np.sin(2 * np.pi * base_freq * t)
    
    # Overtones with decaying amplitude
    for idx, harm in enumerate(harmonics, start=2):
        amp = 0.5 / idx
        phase_jitter = np.random.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * harm * t + phase_jitter)

    # Add a touch of background hiss (real AC lines have thermal noise)
    signal += 0.05 * np.random.randn(num_samples)

    return (signal / np.max(np.abs(signal))).astype(np.float32)


def generate_babble_noise(
    speech_paths: List[str],
    num_samples: int,
    sr: int = 16000,
    num_speakers: int = 4
) -> np.ndarray:
    """
    Simulate cocktail party / cafeteria multi-speaker babble by layering
    multiple distinct human speech recordings with random time shifts.
    """
    if not speech_paths:
        return generate_pink_noise(num_samples)

    selected_files = np.random.choice(speech_paths, size=min(num_speakers, len(speech_paths)), replace=True)
    babble = np.zeros(num_samples, dtype=np.float32)

    for path in selected_files:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        if len(audio) < num_samples:
            repeats = int(np.ceil(num_samples / len(audio)))
            audio = np.tile(audio, repeats)[:num_samples]
        else:
            offset = np.random.randint(0, len(audio) - num_samples + 1)
            audio = audio[offset:offset + num_samples]

        # Randomize volume level of each speaker
        weight = np.random.uniform(0.5, 1.0)
        babble += weight * audio

    if np.max(np.abs(babble)) > 1e-6:
        babble = babble / np.max(np.abs(babble))

    return babble.astype(np.float32)


def create_default_noise_library(
    clean_speech_dir: str,
    output_noise_dir: str,
    sr: int = 16000,
    duration_sec: float = 15.0
) -> List[str]:
    """
    Synthesize and save a standard suite of noise WAV files into output_noise_dir:
    - white_noise.wav
    - pink_noise.wav
    - electrical_hum_60hz.wav
    - cafeteria_babble.wav

    Returns:
        List of generated file paths.
    """
    os.makedirs(output_noise_dir, exist_ok=True)
    num_samples = int(sr * duration_sec)
    clean_files = glob.glob(os.path.join(clean_speech_dir, "*.wav"))

    noise_types = {
        "white_noise.wav": generate_white_noise(num_samples),
        "pink_noise.wav": generate_pink_noise(num_samples),
        "electrical_hum_60hz.wav": generate_hum_noise(num_samples, sr=sr, base_freq=60.0),
        "cafeteria_babble.wav": generate_babble_noise(clean_files, num_samples, sr=sr, num_speakers=5),
    }

    generated_paths = []
    for filename, noise_data in noise_types.items():
        out_path = os.path.join(output_noise_dir, filename)
        sf.write(out_path, noise_data, sr, subtype='PCM_16')
        generated_paths.append(out_path)

    return generated_paths
