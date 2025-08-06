#!/usr/bin/env python3
"""Training script for the NALM dataset."""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from config.train_nalm import *
from data.nalm.streaming import create_nalm_dataloaders, evaluate_nalm_model
from models import GPT, GPTConfig


def get_batch(data_loader, device):
    """Get a batch from the data loader."""
    batch = next(iter(data_loader))
    tokens = batch["tokens"].to(device)
    targets = batch["result"].to(device)
    return tokens, targets


def estimate_loss(model, data_loader, device, eval_iters):
    """Estimate loss on validation data."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        try:
            tokens, targets = get_batch(data_loader, device)
            with torch.no_grad():
                # For NALM, we want to predict the result directly
                # This is a simplified approach - you might want to modify the model output
                logits = model(tokens)
                # Assume the last token contains the result prediction
                result_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

                # Convert targets to token indices (simplified)
                # In practice, you'd need a more sophisticated approach
                target_tokens = (
                    torch.round(targets * 1000).long().clamp(0, vocab_size - 1)
                )

                loss = F.cross_entropy(result_logits, target_tokens)
                losses[k] = loss.item()
        except StopIteration:
            break
    model.train()
    return losses.mean()


def main():
    """Main training function."""
    print("🚀 Starting NALM training...")

    # Set up device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")

    # Create NALM dataloaders
    print("📊 Creating NALM dataloaders...")
    train_dataloader, val_dataloader = create_nalm_dataloaders(
        train_range=nalm_config["train_range"],
        val_range=nalm_config["val_range"],
        extrapolation_range=nalm_config["extrapolation_range"],
        operations=nalm_config["operations"],
        batch_size=nalm_config["batch_size"],
        train_examples=nalm_config["train_examples"],
        val_examples=nalm_config["val_examples"],
        seed=nalm_config["seed"],
    )

    # Create model
    print("🤖 Creating model...")
    if use_dag:
        # Use DAG-GPT model
        config = GPTConfig(
            model_type=model_type,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            dropout=dropout,
            use_dag=True,
            dag_depth=dag_config["dag_depth"],
            max_digits=dag_config["max_digits"],
            max_decimal_places=dag_config["max_decimal_places"],
        )
    else:
        # Use standard GPT model
        config = GPTConfig(
            model_type=model_type,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            dropout=dropout,
        )

    model = GPT(config)
    model = model.to(device)

    # Print model parameters
    print(f"📈 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    # Set up learning rate scheduler
    if decay_lr:

        def get_lr(it):
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            if it > lr_decay_iters:
                return min_lr
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (learning_rate - min_lr)

    else:

        def get_lr(it):
            return learning_rate

    # Set up Weights & Biases
    if wandb_log:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_type": model_type,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "block_size": block_size,
                "learning_rate": learning_rate,
                "max_iters": max_iters,
                "nalm_config": nalm_config,
                "use_dag": use_dag,
            },
        )

    # Training loop
    print("🎯 Starting training loop...")
    model.train()
    iter_num = 0
    best_val_loss = float("inf")

    # Get initial validation loss
    val_loss = estimate_loss(model, val_dataloader, device, eval_iters)
    print(f"📊 Initial validation loss: {val_loss:.4f}")

    while iter_num < max_iters:
        # Determine learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        try:
            tokens, targets = get_batch(train_dataloader, device)
        except StopIteration:
            # Restart the dataloader if it's exhausted
            train_dataloader, _ = create_nalm_dataloaders(
                train_range=nalm_config["train_range"],
                val_range=nalm_config["val_range"],
                extrapolation_range=nalm_config["extrapolation_range"],
                operations=nalm_config["operations"],
                batch_size=nalm_config["batch_size"],
                train_examples=nalm_config["train_examples"],
                val_examples=nalm_config["val_examples"],
                seed=nalm_config["seed"] + iter_num,  # Different seed for each restart
            )
            tokens, targets = get_batch(train_dataloader, device)

        # Forward pass
        logits = model(tokens)

        # For NALM, we want to predict the result directly
        # This is a simplified approach - you might want to modify the model output
        result_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        # Convert targets to token indices (simplified)
        # In practice, you'd need a more sophisticated approach
        target_tokens = (
            torch.round(targets * 1000).long().clamp(0, config.vocab_size - 1)
        )

        loss = F.cross_entropy(result_logits, target_tokens)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Logging
        if iter_num % log_interval == 0:
            print(
                f"📝 Iteration {iter_num}/{max_iters}: loss = {loss.item():.4f}, lr = {lr:.6f}"
            )

            if wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train_loss": loss.item(),
                        "lr": lr,
                    }
                )

        # Evaluation
        if iter_num % eval_interval == 0:
            val_loss = estimate_loss(model, val_dataloader, device, eval_iters)
            print(f"📊 Validation loss: {val_loss:.4f}")

            if wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "val_loss": val_loss,
                    }
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if always_save_checkpoint:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": config,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "nalm_config": nalm_config,
                    }
                    checkpoint_path = out_dir / f"nalm_best.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                    # Remove previous best checkpoint if it exists
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        print(
                            f"🗑️ Removed previous best checkpoint: {checkpoint_path.name}"
                        )

                    torch.save(checkpoint, checkpoint_path)
                    print(f"💾 Saved best model to {checkpoint_path}")

        iter_num += 1

    print("✅ Training completed!")

    # Final evaluation
    print("📊 Running final evaluation...")
    final_metrics = evaluate_nalm_model(model, val_dataloader, device)
    print(f"Final accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final MSE: {final_metrics['mse']:.6f}")

    if wandb_log:
        wandb.log(
            {
                "final_accuracy": final_metrics["accuracy"],
                "final_mse": final_metrics["mse"],
            }
        )
        wandb.finish()


if __name__ == "__main__":
    import math

    main()
