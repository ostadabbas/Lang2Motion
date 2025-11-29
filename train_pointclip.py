#!/usr/bin/env python3
"""
PointCLIP Training Script
Trains MotionCLIP with point trajectories from MeViS dataset using CoTracker3
"""

import os
import sys
sys.path.append('.')

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.train.trainer import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data


def do_epochs(model, datasets, parameters, optimizer, writer):
    """Training loop for PointCLIP"""
    dataset = datasets["train"]
    train_iterator = DataLoader(
        dataset, 
        batch_size=parameters["batch_size"],
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid CUDA multiprocessing issues
        collate_fn=collate
    )

    # Print active losses at start of training
    print("\n" + "="*60)
    print("🎯 ACTIVE LOSS CONFIGURATION:")
    print("="*60)
    
    # Print main losses
    lambdas = model.lambdas
    for loss_name, loss_weight in lambdas.items():
        # Handle both simple floats and dicts
        if isinstance(loss_weight, dict):
            # Skip complex loss structures for now
            continue
        if loss_weight > 0:
            print(f"  ✅ {loss_name}: λ = {loss_weight}")
        else:
            print(f"  ❌ {loss_name}: λ = {loss_weight} (DISABLED)")
    
    # Print CLIP losses if present
    if hasattr(model, 'clip_lambdas') and model.clip_lambdas:
        print("\n  CLIP Losses:")
        for loss_name, loss_value in model.clip_lambdas.items():
            # CLIP lambdas might be dicts with multiple sub-losses
            if isinstance(loss_value, dict):
                for sub_name, sub_weight in loss_value.items():
                    if sub_weight > 0:
                        print(f"  ✅ clip_{loss_name}_{sub_name}: λ = {sub_weight}")
            else:
                if loss_value > 0:
                    print(f"  ✅ clip_{loss_name}: λ = {loss_value}")
    
    print("="*60 + "\n")

    # Get start epoch (0 if training from scratch, checkpoint epoch if resuming)
    start_epoch = parameters.get('start_epoch', 0)
    
    logpath = os.path.join(parameters["folder"], "training.log")
    mode = "a" if start_epoch > 0 else "w"  # Append if resuming, write if new
    with open(logpath, mode) as logfile:
        for epoch in range(start_epoch + 1, parameters["num_epochs"]+1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{parameters['num_epochs']}")
            print(f"{'='*60}")
            
            dict_loss = train(model, optimizer, train_iterator, model.device)

            # After backward() inside train(), we can print grad for position_scale if present
            try:
                dec = getattr(model, 'decoder', None)
                if dec is not None and hasattr(dec, 'position_scale'):
                    grad = dec.position_scale.grad
                    print(f"[GRAD_CHECK] position_scale={dec.position_scale.item():.6f}, grad={(grad.item() if grad is not None else 'None')}")
            except Exception:
                pass

            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                
                # Save without CLIP model (as done in original training script)
                if parameters.get('clip_training', '') == '':
                    state_dict_wo_clip = {k: v for k,v in model.state_dict().items() 
                                         if not k.startswith('clip_model.')}
                else:
                    state_dict_wo_clip = model.state_dict()
                torch.save(state_dict_wo_clip, checkpoint_path)

            writer.flush()


def main():
    """Main training function"""
    print("Starting PointCLIP Training")
    print("="*60)
    
    # Parse command line arguments
    parameters = parser()
    
    # Print key parameters
    print("Training Configuration:")
    print(f"  Dataset: {parameters['dataset']}")
    print(f"  Data path: {parameters.get('data_path', 'default')}")
    print(f"  Grid size: {parameters.get('grid_size', 16)} ({parameters.get('grid_size', 16)**2} points)")
    print(f"  Num frames: {parameters['num_frames']}")
    print(f"  Batch size: {parameters['batch_size']}")
    print(f"  Learning rate: {parameters['lr']}")
    print(f"  Epochs: {parameters['num_epochs']}")
    print(f"  Latent dim: {parameters['latent_dim']}")
    print(f"  Device: {parameters['device']}")
    print(f"  Output folder: {parameters['folder']}")
    
    # Logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])

    device = parameters["device"]
    print(f"\nUsing device: {device}")

    # Load model and datasets
    print("\nLoading model and datasets...")
    model, datasets = get_model_and_data(parameters)

    # CRITICAL: Verify decoder type
    print("\n" + "="*60)
    print("🔍 DECODER TYPE VERIFICATION:")
    print("="*60)
    decoder = model.decoder
    decoder_type = type(decoder).__name__
    print(f"  Decoder class: {decoder_type}")
    
    if 'MLP' in decoder_type:
        print("  ⚠️  WARNING: Using MLP Decoder!")
        print("     If you wanted Transformer, check --mlp_decoder flag!")
    elif 'TRANSFORMER' in decoder_type:
        print("  ✅ CORRECT: Using Transformer Decoder")
        # Verify it has transformer layers
        if hasattr(decoder, 'seqTransEncoder'):
            num_layers = len(decoder.seqTransEncoder.layers)
            print(f"     Transformer layers: {num_layers}")
        else:
            print("  ⚠️  WARNING: Transformer decoder missing seqTransEncoder!")
    else:
        print(f"  ⚠️  UNKNOWN decoder type: {decoder_type}")
    print("="*60 + "\n")

    # Print actual lambda values used in model
    print("\n=== ACTUAL LAMBDAS ===")
    if hasattr(model, 'lambdas'):
        print(f"model.lambdas = {getattr(model, 'lambdas')}")
    if hasattr(model, 'clip_lambdas'):
        print(f"model.clip_lambdas = {getattr(model, 'clip_lambdas')}")
    print("=" * 60)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])
    
    # Learning rate scheduler: Step decay every 50 epochs
    # Start: 0.0001 → 0.00005 (epoch 50) → 0.00001 (epoch 100) → 0.000005 (epoch 150)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,  # Decay every 50 epochs
        gamma=0.5      # Reduce by 50%
    )
    print(f"✅ Learning rate scheduler: StepLR(step_size=50, gamma=0.5)")
    print(f"   Initial LR: {parameters['lr']}")
    print(f"   Will decay to: {parameters['lr'] * 0.5} at epoch 50, {parameters['lr'] * 0.25} at epoch 100")
    
    # Load checkpoint if specified
    start_epoch = 0
    if parameters.get('checkpoint') is not None:
        checkpoint_path = parameters['checkpoint']
        if os.path.exists(checkpoint_path):
            print(f"\n🔄 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint, strict=False)
            print("✅ Model weights loaded")
            
            # Extract epoch number from checkpoint name (e.g., checkpoint_0165.pth.tar -> 165)
            import re
            match = re.search(r'checkpoint_(\d+)\.pth\.tar', checkpoint_path)
            if match:
                start_epoch = int(match.group(1))
                print(f"✅ Resuming from epoch {start_epoch}")
                
                # CRITICAL: Smart LR reset based on epoch number
                # At high epochs, LR has decayed too much - reset to a reasonable value
                # This prevents over-decay while maintaining training stability
                initial_lr = parameters['lr']
                
                # Calculate what LR would be at this epoch with normal decay
                steps_decayed = start_epoch // 50
                decayed_lr = initial_lr * (0.5 ** steps_decayed)
                
                # Smart reset: use higher LR if we've decayed too much
                if start_epoch >= 300:
                    # Very late training: use moderate LR to continue learning
                    reset_lr = initial_lr * 0.25  # 0.000025
                    print(f"🔄 Late training detected (epoch {start_epoch})")
                    print(f"   Decayed LR would be: {decayed_lr:.8f} (too low!)")
                    print(f"   Resetting to: {reset_lr:.6f} (moderate restart)")
                elif start_epoch >= 200:
                    # Mid-late training: use half LR
                    reset_lr = initial_lr * 0.5  # 0.00005
                    print(f"🔄 Mid-late training detected (epoch {start_epoch})")
                    print(f"   Decayed LR would be: {decayed_lr:.8f} (too low!)")
                    print(f"   Resetting to: {reset_lr:.6f} (moderate restart)")
                else:
                    # Early training: use full LR
                    reset_lr = initial_lr
                    print(f"🔄 Early training (epoch {start_epoch})")
                    print(f"   Resetting to initial LR: {reset_lr:.6f}")
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = reset_lr
                print(f"✅ Learning rate set to: {reset_lr:.6f}")
                print(f"   (Warm restart: model can learn effectively again)")
                
                # Reset scheduler to start from beginning (as if starting fresh)
                # This allows proper decay schedule from the resume point
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=50,
                    gamma=0.5
                )
                print(f"✅ Scheduler reset (will decay from epoch {start_epoch+1})")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            print("Starting from scratch...")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    
    print(f"\nModel Information:")
    print(f"  Total params: {total_params:.2f}M")
    print(f"  Trainable params: {trainable_params:.2f}M")
    print(f"  Dataset size: {len(datasets['train'])}")
    
    # Verify dataset properties
    sample = datasets['train'][0]
    if sample is not None:
        print(f"  Sample input shape: {sample['inp'].shape}")
        print(f"  Expected: [{parameters.get('grid_size', 16)**2}, 2, {parameters['num_frames']}]")
    
    print("\nStarting training...")
    parameters['start_epoch'] = start_epoch
    parameters['scheduler'] = scheduler  # Pass scheduler to training loop
    do_epochs(model, datasets, parameters, optimizer, writer)

    writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
