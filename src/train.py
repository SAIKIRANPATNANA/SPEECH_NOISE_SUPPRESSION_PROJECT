"""
train.py - PyTorch Training Pipeline for Audio U-Net Noise Suppression
======================================================================

A CSE Guide to Training Speech Enhancement Models:
--------------------------------------------------
1. What Loss Function should we use?
   - In computer vision, L1/L2 loss measures pixel differences.
   - In Audio Spectrograms:
     a) Magnitude L1 Loss: || |S| - |S_hat| ||_1
        Measures exact spectral energy error. L1 is preferred over L2/MSE because
        L2 squares errors, causing the network to ignore quiet phonemes (like 'th', 'f', 'p')
        in favor of loud vowel formants.
     b) Mask Loss (MSE / BCE): || M_target - M_pred ||^2
        Supervises the filter mask directly.
     Combined Loss: L = L_mag + 0.5 * L_mask gives the best stability and speech clarity!

2. Optimizer & Learning Rate:
   Adam optimizer with lr=1e-3, beta1=0.9, beta2=0.999.
   Learning rate scheduler (ReduceLROnPlateau) halves the learning rate when validation
   loss plateaus, allowing delicate fine-tuning.
"""

import os
import sys
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import create_dataloaders
from src.model import UNetSpeechEnhancer


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion_mag: nn.Module,
    criterion_mask: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Train the model for a single epoch over all mini-batches."""
    model.train()
    total_loss = 0.0
    total_mag_loss = 0.0
    total_mask_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        noisy_mag = batch["noisy_mag"].to(device)   # [B, 1, 256, 256]
        clean_mag = batch["clean_mag"].to(device)   # [B, 1, 256, 256]
        irm_mask = batch["irm_mask"].to(device)     # [B, 1, 256, 256]

        optimizer.zero_grad()

        # Forward pass
        pred_clean_mag, pred_mask = model(noisy_mag)

        # Compute losses
        loss_mag = criterion_mag(pred_clean_mag, clean_mag)
        
        if pred_mask is not None:
            loss_mask = criterion_mask(pred_mask, irm_mask)
            loss = loss_mag + 0.5 * loss_mask
            total_mask_loss += loss_mask.item()
        else:
            loss = loss_mag

        # Backpropagation
        loss.backward()
        # Gradient clipping to prevent gradient explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_mag_loss += loss_mag.item()

    return {
        "loss": total_loss / num_batches,
        "mag_loss": total_mag_loss / num_batches,
        "mask_loss": total_mask_loss / num_batches if pred_mask is not None else 0.0
    }


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion_mag: nn.Module,
    criterion_mask: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model performance on the held-out validation set."""
    model.eval()
    total_loss = 0.0
    total_mag_loss = 0.0
    total_mask_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            noisy_mag = batch["noisy_mag"].to(device)
            clean_mag = batch["clean_mag"].to(device)
            irm_mask = batch["irm_mask"].to(device)

            pred_clean_mag, pred_mask = model(noisy_mag)

            loss_mag = criterion_mag(pred_clean_mag, clean_mag)
            
            if pred_mask is not None:
                loss_mask = criterion_mask(pred_mask, irm_mask)
                loss = loss_mag + 0.5 * loss_mask
                total_mask_loss += loss_mask.item()
            else:
                loss = loss_mag

            total_loss += loss.item()
            total_mag_loss += loss_mag.item()

    return {
        "loss": total_loss / num_batches,
        "mag_loss": total_mag_loss / num_batches,
        "mask_loss": total_mask_loss / num_batches if pred_mask is not None else 0.0
    }


def run_training(
    clean_dir: str = "data/clean_speech",
    noise_dir: str = "data/noise",
    output_dir: str = "models",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_split: float = 0.2,
    mode: str = "masking",
    device_str: Optional[str] = None
) -> str:
    """
    Full training routine with checkpointing and metric history logging.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine hardware acceleration
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Training] Using compute device: {device}")

    # 1. Prepare DataLoaders
    print(f"[Training] Loading datasets from clean='{clean_dir}' and noise='{noise_dir}'...")
    train_loader, val_loader = create_dataloaders(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=2 if os.name != 'nt' else 0
    )
    print(f"[Training] Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # 2. Instantiate Model
    model = UNetSpeechEnhancer(mode=mode).to(device)
    print(f"[Training] Initialized UNetSpeechEnhancer ({mode} mode) with {model.get_num_parameters():,} parameters.")

    # 3. Optimizers & Loss Functions
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    criterion_mag = nn.L1Loss()
    criterion_mask = nn.MSELoss()

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "best_model.pt")
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_mag_loss": [], "val_mag_loss": [],
        "lr": []
    }

    start_time = time.time()
    print("\n" + "=" * 70)
    print(f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'Val Mag L1':<15}{'Time':<10}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion_mag, criterion_mask, device
        )
        val_metrics = validate(
            model, val_loader, criterion_mag, criterion_mask, device
        )
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_mag_loss"].append(train_metrics["mag_loss"])
        history["val_mag_loss"].append(val_metrics["mag_loss"])
        history["lr"].append(current_lr)

        ep_duration = time.time() - ep_start
        print(f"{epoch:<8}{train_metrics['loss']:<15.4f}{val_metrics['loss']:<15.4f}{val_metrics['mag_loss']:<15.4f}{ep_duration:<8.1f}s")

        # Save checkpoint if best validation score achieved
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "mode": mode
            }, best_checkpoint_path)

    total_duration = time.time() - start_time
    print("=" * 70)
    print(f"[Training Complete] Best Val Loss: {best_val_loss:.4f} | Total Time: {total_duration/60:.2f} mins")
    print(f"[Saved Checkpoint] -> {best_checkpoint_path}")

    # Save training history JSON
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Audio Noise Suppression Model")
    parser.add_argument("--clean_dir", type=str, default="data/clean_speech", help="Path to clean speech WAVs")
    parser.add_argument("--noise_dir", type=str, default="data/noise", help="Path to noise WAVs")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--mode", type=str, default="masking", choices=["masking", "mapping"], help="Model mode")
    args = parser.parse_args()

    run_training(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mode=args.mode
    )
