"""
inference.py - Full-Length Speech Enhancement Engine & CLI
==========================================================

A CSE Guide to Audio Enhancement Inference:
-------------------------------------------
1. The Problem with Arbitrary Length Audio:
   During training, we used fixed 256x256 spectrogram crops (~2.05s).
   In real life, a user might pass an audio clip that is 1.4 seconds, 5.2 seconds,
   or 30 seconds long.

2. The Fully-Convolutional Secret:
   Our U-Net contains ONLY 2D convolutions (no hard-coded linear / dense layers).
   Downsampling depth = 5 stages (Enc1, Enc2, Enc3, Enc4, Bottleneck) -> 2^5 = 32.
   Therefore, as long as the time dimension T is padded to a multiple of 32:
     - The U-Net can process the ENTIRE audio spectrogram in a SINGLE forward pass!
     - Zero chunk boundary artifacts, zero seam clicks, zero phase discontinuites.
     - After prediction, we simply slice off the padded time frames back to original T.

3. Reconstructing the Audio Waveform:
   a) Clean Magnitude Prediction: S_hat = Mask * Y_mag
   b) Restore Nyquist Bin: Add back the 257th frequency bin (zeroed) to match 512-point FFT.
   c) Combine with Noisy Phase: Z_enhanced = S_hat * exp(j * Phase_noisy)
   d) Inverse STFT: Overlap-Add synthesis produces the enhanced 1D waveform!
"""

import os
import sys
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch

from src.audio_processing import (
    load_audio,
    save_audio,
    compute_stft,
    reconstruct_waveform,
    mag_to_db
)
from src.model import UNetSpeechEnhancer
from src.evaluation import (
    calculate_snr,
    calculate_si_sdr,
    calculate_lsd,
    plot_waveform_comparison,
    plot_spectrogram_comparison
)


def load_model(checkpoint_path: str, device: torch.device) -> UNetSpeechEnhancer:
    """Load trained U-Net checkpoint from disk."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    mode = checkpoint.get("mode", "masking")
    model = UNetSpeechEnhancer(mode=mode).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def enhance_audio(
    model: UNetSpeechEnhancer,
    noisy_waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhance an arbitrary-length noisy audio waveform using the trained U-Net.

    Parameters:
        model: Trained UNetSpeechEnhancer instance
        noisy_waveform: 1D numpy array of noisy audio
        sr: Sampling rate (16000 Hz)
        device: Torch compute device

    Returns:
        enhanced_audio (np.ndarray): 1D enhanced audio waveform
        noisy_mag (np.ndarray): Full noisy magnitude spectrogram
        enhanced_mag (np.ndarray): Full enhanced magnitude spectrogram
        predicted_mask (np.ndarray): Full predicted attenuation mask
    """
    if device is None:
        device = next(model.parameters()).device

    orig_length = len(noisy_waveform)

    # 1. Compute STFT (Magnitude & Phase)
    noisy_mag_full, noisy_phase_full = compute_stft(
        noisy_waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    F_full, T_full = noisy_mag_full.shape

    # 2. Prepare 256 frequency bins (drop 257th Nyquist bin)
    freq_input = 256
    noisy_mag_256 = noisy_mag_full[:freq_input, :]

    # 3. Time dimension padding to nearest multiple of 32 (for 5-stage U-Net)
    stride_multiple = 32
    pad_t = 0
    if T_full % stride_multiple != 0:
        pad_t = stride_multiple - (T_full % stride_multiple)
    
    # Ensure at least 32 frames
    if T_full + pad_t < 32:
        pad_t = 32 - T_full

    if pad_t > 0:
        noisy_mag_padded = np.pad(noisy_mag_256, ((0, 0), (0, pad_t)), mode='reflect')
    else:
        noisy_mag_padded = noisy_mag_256

    # 4. Neural Network Inference
    tensor_in = torch.from_numpy(noisy_mag_padded).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        enh_mag_tensor, mask_tensor = model(tensor_in)
        enh_mag_256 = enh_mag_tensor.squeeze().cpu().numpy()
        if mask_tensor is not None:
            mask_256 = mask_tensor.squeeze().cpu().numpy()
        else:
            mask_256 = np.ones_like(enh_mag_256)

    # 5. Remove time padding
    enh_mag_unpad = enh_mag_256[:, :T_full]
    mask_unpad = mask_256[:, :T_full]

    # 6. Restore 257th frequency bin for full-spectrum iSTFT
    enh_mag_full = np.zeros((F_full, T_full), dtype=np.float32)
    enh_mag_full[:freq_input, :] = enh_mag_unpad
    # Nyquist bin energy is negligible; set to zero or copy top bin
    enh_mag_full[freq_input:, :] = 0.0

    mask_full = np.zeros((F_full, T_full), dtype=np.float32)
    mask_full[:freq_input, :] = mask_unpad
    mask_full[freq_input:, :] = 0.0

    # 7. Reconstruct waveform via iSTFT + Noisy Phase
    enhanced_audio = reconstruct_waveform(
        enh_mag_full,
        noisy_phase_full,
        hop_length=hop_length,
        win_length=win_length,
        length=orig_length
    )

    # Peak normalize output audio
    max_val = np.max(np.abs(enhanced_audio))
    if max_val > 1e-6:
        enhanced_audio = enhanced_audio / max_val

    return enhanced_audio, noisy_mag_full, enh_mag_full, mask_full


def enhance_file(
    checkpoint_path: str,
    input_wav_path: str,
    output_wav_path: str,
    clean_reference_path: Optional[str] = None,
    plot_save_path: Optional[str] = None,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level function to enhance a single audio file and optionally evaluate metrics.
    """
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint_path, device)

    noisy_audio, sr = load_audio(input_wav_path, target_sr=16000)
    enhanced_audio, noisy_mag, enh_mag, mask = enhance_audio(model, noisy_audio, sr=sr, device=device)

    save_audio(output_wav_path, enhanced_audio, sr=sr)

    results: Dict[str, Any] = {
        "input_wav": input_wav_path,
        "output_wav": output_wav_path,
        "duration_sec": len(noisy_audio) / sr
    }

    if clean_reference_path and os.path.exists(clean_reference_path):
        clean_audio, _ = load_audio(clean_reference_path, target_sr=16000)
        clean_mag, _ = compute_stft(clean_audio)

        snr_in = calculate_snr(clean_audio, noisy_audio)
        snr_out = calculate_snr(clean_audio, enhanced_audio)
        delta_snr = snr_out - snr_in

        sdr_in = calculate_si_sdr(clean_audio, noisy_audio)
        sdr_out = calculate_si_sdr(clean_audio, enhanced_audio)
        delta_sdr = sdr_out - sdr_in

        lsd_in = calculate_lsd(clean_mag, noisy_mag)
        lsd_out = calculate_lsd(clean_mag, enh_mag)

        results.update({
            "snr_input_db": snr_in,
            "snr_output_db": snr_out,
            "delta_snr_db": delta_snr,
            "si_sdr_input_db": sdr_in,
            "si_sdr_output_db": sdr_out,
            "delta_si_sdr_db": delta_sdr,
            "lsd_input_db": lsd_in,
            "lsd_output_db": lsd_out
        })

        if plot_save_path:
            plot_spectrogram_comparison(
                clean_mag, noisy_mag, enh_mag, mask, sr=sr, save_path=plot_save_path
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance Noisy Audio with Trained U-Net")
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input noisy WAV file")
    parser.add_argument("--output", type=str, required=True, help="Output enhanced WAV file")
    parser.add_argument("--clean", type=str, default=None, help="Optional clean reference WAV for metrics")
    parser.add_argument("--plot", type=str, default=None, help="Path to save comparison plot PNG")
    args = parser.parse_args()

    res = enhance_file(
        checkpoint_path=args.model,
        input_wav_path=args.input,
        output_wav_path=args.output,
        clean_reference_path=args.clean,
        plot_save_path=args.plot
    )
    print("\n--- Enhancement Results ---")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
