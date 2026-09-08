"""
audio_processing.py - Digital Signal Processing (DSP) Core Utilities
====================================================================

A CSE Guide to Audio Processing:
--------------------------------
1. Audio Waveform (Time Domain):
   An audio file is simply an array of floats (e.g., float[32000] for 2 seconds at 16kHz).
   Each float represents instantaneous air pressure at that microsecond.
   Problem: Speech and noise are mixed additively (y[t] = s[t] + n[t]).
   In 1D, it is impossible to separate which part of the float belongs to voice vs noise.

2. Short-Time Fourier Transform (STFT):
   STFT acts like an "audio decompiler". It slices the 1D audio array into overlapping
   time windows (frames) and applies the Discrete Fourier Transform (DFT) to each.
   Result: A 2D matrix of complex numbers Z[f, t] where:
     - Rows (f): Frequency bins (pitch, from bass to treble).
     - Columns (t): Time steps (frames).
   
3. Magnitude vs Phase:
   Every complex number can be written as Z = Magnitude * exp(j * Phase):
     - Magnitude (|Z|): How much energy/loudness exists at frequency f and time t.
       (The human ear is sensitive to this! It contains speech formants and vowels).
     - Phase (angle(Z)): The exact cycle offset / alignment within that millisecond.
       (The human ear is mostly insensitive to phase for intelligibility).
   
4. Inverse STFT (iSTFT):
   Overlap-Add (OLA) synthesis stitches the frequency frames back into a continuous
   1D time-domain audio wave using smooth Hann window blending.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
import soundfile as sf
import librosa
import torch


def load_audio(
    file_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample to target_sr, convert to mono, and normalize.

    Parameters:
        file_path (str): Path to the audio file (.wav, .flac, .mp3, etc.)
        target_sr (int): Target sampling rate in Hz (default: 16,000 Hz)
        mono (bool): Convert multi-channel audio to single-channel (mono)
        normalize (bool): Peak-normalize signal into range [-1.0, 1.0]

    Returns:
        audio (np.ndarray): 1D float32 numpy array of audio samples
        sr (int): Sampling rate
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Use librosa to load and automatically resample
    audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)

    if normalize and len(audio) > 0:
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val

    return audio.astype(np.float32), sr


def save_audio(
    file_path: str,
    audio: Union[np.ndarray, torch.Tensor],
    sr: int = 16000
) -> None:
    """
    Save an audio waveform to disk as a 16-bit PCM WAV file.

    Parameters:
        file_path (str): Output file path (.wav)
        audio (np.ndarray | torch.Tensor): 1D audio samples
        sr (int): Sampling rate in Hz
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    audio = np.squeeze(audio)
    
    # Clip to prevent digital clipping / distortion
    audio = np.clip(audio, -1.0, 1.0)

    # Save using soundfile (clean 16-bit PCM)
    sf.write(file_path, audio, sr, subtype='PCM_16')


def compute_stft(
    waveform: Union[np.ndarray, torch.Tensor],
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal.
    Decomposes 1D waveform into 2D Magnitude and Phase matrices.

    Parameters:
        waveform (np.ndarray | torch.Tensor): 1D audio array [samples] or [batch, samples]
        n_fft (int): Number of FFT points (defines frequency resolution; 512 gives 257 bins)
        hop_length (int): Number of samples between frames (128 samples @ 16kHz = 8ms)
        win_length (int): Window size (default: 512 samples = 32ms)

    Returns:
        magnitude: Absolute energy spectrum |Z|
        phase: Angle in radians [-pi, pi]
    """
    if isinstance(waveform, torch.Tensor):
        # PyTorch STFT
        device = waveform.device
        window = torch.hann_window(win_length, device=device)
        
        # Ensure 2D tensor [batch, samples] or [1, samples]
        is_1d = waveform.dim() == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)
            
        stft_complex = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        
        magnitude = torch.abs(stft_complex)
        phase = torch.angle(stft_complex)
        
        if is_1d:
            magnitude = magnitude.squeeze(0)
            phase = phase.squeeze(0)
            
        return magnitude, phase

    else:
        # NumPy / Librosa STFT
        stft_complex = librosa.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window='hann'
        )
        magnitude = np.abs(stft_complex)
        phase = np.angle(stft_complex)
        return magnitude.astype(np.float32), phase.astype(np.float32)


def reconstruct_waveform(
    magnitude: Union[np.ndarray, torch.Tensor],
    phase: Union[np.ndarray, torch.Tensor],
    hop_length: int = 128,
    win_length: int = 512,
    length: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reconstruct the 1D time-domain audio waveform from Magnitude and Phase using iSTFT.
    Formula: Z = Magnitude * exp(j * Phase)
             waveform = iSTFT(Z) via Overlap-Add (OLA).

    Parameters:
        magnitude: 2D magnitude spectrogram |Z|
        phase: 2D phase matrix angle(Z) (e.g., the original noisy phase)
        hop_length: Hop length matching forward STFT
        win_length: Window length matching forward STFT
        length: Expected length in samples (for exact truncation/padding)

    Returns:
        reconstructed_audio: 1D audio waveform
    """
    if isinstance(magnitude, torch.Tensor):
        device = magnitude.device
        window = torch.hann_window(win_length, device=device)
        
        # Recombine into complex tensor
        complex_stft = torch.polar(magnitude, phase)
        
        is_2d = complex_stft.dim() == 2
        if is_2d:
            complex_stft = complex_stft.unsqueeze(0)
            
        n_fft = (magnitude.shape[-2] - 1) * 2
        waveform = torch.istft(
            complex_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=length
        )
        
        if is_2d:
            waveform = waveform.squeeze(0)
            
        return waveform

    else:
        # NumPy / Librosa iSTFT
        complex_stft = magnitude * np.exp(1j * phase)
        waveform = librosa.istft(
            complex_stft,
            hop_length=hop_length,
            win_length=win_length,
            window='hann',
            length=length
        )
        return waveform.astype(np.float32)


def pad_or_crop_spectrogram(
    spectrogram: np.ndarray,
    target_frames: int = 256,
    target_freq: int = 256
) -> np.ndarray:
    """
    Crop or pad a 2D spectrogram matrix to exact [target_freq, target_frames] dimensions.
    
    Why this is crucial for U-Net in CSE:
    Convolutional networks with 4 downsampling stages divide dimensions by 2^4 = 16.
    If input is 256x256, dimensions scale cleanly: 256 -> 128 -> 64 -> 32 -> 16.
    An STFT with n_fft=512 naturally has 257 frequency bins. Dropping bin 256 (Nyquist frequency
    at 8000 Hz, where speech has zero energy) gives a pristine 256x256 matrix!

    Parameters:
        spectrogram (np.ndarray): 2D array of shape [F, T]
        target_frames (int): Number of time frames (columns)
        target_freq (int): Number of frequency bins (rows)

    Returns:
        processed_spectrogram (np.ndarray): Shape [target_freq, target_frames]
    """
    F, T = spectrogram.shape

    # 1. Frequency cropping / padding
    if F >= target_freq:
        # Keep lowest target_freq bins (speech energy is in lower frequencies)
        spec = spectrogram[:target_freq, :]
    else:
        pad_f = target_freq - F
        spec = np.pad(spectrogram, ((0, pad_f), (0, 0)), mode='constant')

    # 2. Time frames cropping / padding
    if T >= target_frames:
        spec = spec[:, :target_frames]
    else:
        pad_t = target_frames - T
        spec = np.pad(spec, ((0, 0), (0, pad_t)), mode='constant')

    return spec.astype(np.float32)


def mag_to_db(
    magnitude: Union[np.ndarray, torch.Tensor],
    ref: float = 1.0,
    amin: float = 1e-5,
    top_db: float = 80.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert linear magnitude spectrogram to decibels (log-scale) for visualization.
    dB = 20 * log10(magnitude / ref)
    """
    if isinstance(magnitude, torch.Tensor):
        power = torch.clamp(magnitude, min=amin)
        db = 20.0 * torch.log10(power / ref)
        if top_db is not None:
            max_val = torch.max(db)
            db = torch.clamp(db, min=max_val - top_db)
        return db
    else:
        power = np.maximum(magnitude, amin)
        db = 20.0 * np.log10(power / ref)
        if top_db is not None:
            max_val = np.max(db)
            db = np.maximum(db, max_val - top_db)
        return db.astype(np.float32)
