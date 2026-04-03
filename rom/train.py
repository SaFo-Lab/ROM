import os
import math
import argparse
import yaml
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from rom.models import StreamingHead
from rom.dataset import DatasetFromJSONL, collate_fn, LengthBucketSampler
from rom.env import setup_hf_cache, set_seed


def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params / 1_000_000


def plot_training_curves(epoch_losses, epoch_accs, save_dir, loss_suffix, model_name):
    """Plot loss and accuracy trends side by side"""
    epochs = list(range(len(epoch_losses)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, epoch_losses, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, epoch_accs, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (token)', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_curves_{model_name}_{loss_suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {save_path}")
    return save_path





def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = True
    
    if args.use_wandb:
        if args.wandb_run_name is None:
            short_model = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
            _has_aug = any("_aug" in (d or "") for d in [args.efficient_data, args.overthinking_data])
            aug_tag = "_aug" if _has_aug else ""
            args.wandb_run_name = f"{short_model}_L{args.idx_layer}{aug_tag}_ep{args.num_train_epochs}"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=os.environ.get('WANDB_ENTITY', None),
            config=vars(args)
        )
    
    print(f"Loading model: {args.model_name}...")
    use_device_map = torch.cuda.is_available()
    # Use single GPU to avoid device mismatch issues with multi-GPU setups
    if use_device_map:
        device_map = str(device)  # Use "cuda:0" instead of "auto" to force single GPU
    else:
        device_map = None
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map=device_map,
        trust_remote_code=True
    )
    print("Model loaded successfully!")
    
    if not use_device_map:
        base_model = base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print("Tokenizer loaded!")

    # Resolve full paths for data files
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Kelp-my/
    efficient_path = os.path.join(script_dir, args.efficient_data) if args.efficient_data else None
    overthinking_path = os.path.join(script_dir, args.overthinking_data) if args.overthinking_data else None
    dataset_dir = os.path.dirname(efficient_path or overthinking_path)

    # Check if input files contain "_aug" suffix
    has_aug = any("_aug" in str(p) for p in (efficient_path, overthinking_path) if p is not None)
    
    print(f"\nInitializing training dataset from: {dataset_dir}")
    if efficient_path:
        print(f"  Efficient data: {efficient_path}")
    if overthinking_path:
        print(f"  Overthinking data: {overthinking_path}")
    
    # Auto batch cache building if max_build_samples is specified
    if args.max_build_samples is not None:
        import json
        # Count total samples first
        total_samples = 0
        for path in [efficient_path, overthinking_path]:
            if path and os.path.exists(path):
                with open(path, 'r') as f:
                    total_samples += sum(1 for _ in f)
        
        print(f"\n{'='*60}")
        print(f"Auto batch cache building enabled")
        print(f"Total samples: {total_samples}, Batch size: {args.max_build_samples}")
        print(f"{'='*60}\n")
        
        # Build cache in batches
        start_idx = 0
        while start_idx < total_samples:
            batch_num = start_idx // args.max_build_samples + 1
            end_idx = min(start_idx + args.max_build_samples, total_samples)
            print(f"[Batch {batch_num}] Building cache for samples {start_idx}-{end_idx-1}...")
            
            DatasetFromJSONL(
                dataset_dir=dataset_dir,
                model_name=args.model_name,
                tokenizer=tokenizer,
                base_model=base_model,
                idx_layer=args.idx_layer,
                device=device,
                build_cache_if_missing=True,
                overwrite=False,
                max_build_samples=args.max_build_samples,
                start_build_idx=start_idx,
                efficient_data=efficient_path,
                overthinking_data=overthinking_path,
            )
            print(f"✓ Batch {batch_num} completed\n")
            start_idx = end_idx
            
            # Clear cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"{'='*60}")
        print(f"All cache building completed!")
        print(f"{'='*60}\n")
    
    # Load the full dataset (cache already built)
    train_dataset = DatasetFromJSONL(
        dataset_dir=dataset_dir,
        model_name=args.model_name,
        tokenizer=tokenizer,
        base_model=base_model,
        idx_layer=args.idx_layer,
        device=device,
        build_cache_if_missing=True,
        overwrite=False,
        max_build_samples=None,  # Load all cached samples
        start_build_idx=0,
        efficient_data=efficient_path,
        overthinking_data=overthinking_path,
    )
    print(f"Training dataset ready: {len(train_dataset)} samples")

    bucket_sampler = LengthBucketSampler(
        train_dataset.lengths, batch_size=args.batch_size, shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=bucket_sampler,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True, prefetch_factor=4,
    )

    input_dim = AutoConfig.from_pretrained(args.model_name).hidden_size

    del base_model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    head = StreamingHead(
            input_dim=input_dim,
            proj_dim=1024, 
            mem_dim=1024, 
            num_labels=2, 
            use_dt=True,
            cfc=True)


    head.to(device=device, dtype=torch.bfloat16)
    head.requires_grad = True

    print("Total trainable parameters: ", count_parameters(head), 'M')

    if args.use_wandb:
        wandb.config.update({
            "trainable_params_M": count_parameters(head),
            "dataset_size": len(train_dataset),
            "input_dim": input_dim,
        }, allow_val_change=True)
        wandb.watch(head, log="all", log_freq=100)

    start_epoch = 0
    
    # Load checkpoint if resuming
    if args.resume_from_checkpoint is not None:
        print(f"Loading checkpoint from: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            head.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Backward compatibility: old checkpoint only has state_dict
            head.load_state_dict(checkpoint)
            print("Loaded model weights (old format, no optimizer/scheduler state)")
        
        print("Checkpoint loaded successfully!")

    optimizer = AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    max_grad_norm = 1.0

    max_steps = -1
    lr_scheduler_type = "cosine"
    warmup_ratio = args.warmup_ratio if hasattr(args, 'warmup_ratio') else 0.05
    warmup_steps = 0

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_acc_steps)
    if max_steps is None or max_steps < 0:
        total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        total_training_steps = max_steps

    if warmup_steps and warmup_steps > 0:
        computed_warmup_steps = warmup_steps
    else:
        computed_warmup_steps = int(total_training_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=computed_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    # Load optimizer and scheduler state if resuming
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Extract model name for filename (handle both paths and model IDs)
    model_name_for_file = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name

    # Determine checkpoint naming
    suffix_parts = []
    suffix_parts.append(f"L{args.idx_layer}")
    if has_aug:
        suffix_parts.append("aug")
    loss_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    # Initialize or restore step counters
    global_step = 0
    completed_steps = 0
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        if 'completed_steps' in checkpoint:
            completed_steps = checkpoint['completed_steps']
        print(f"Resuming from step {completed_steps}")
    
    head.train()

    # Track training metrics for plotting
    training_history = {
        'losses': [],
        'losses_ce': [],
        'accs': []
    }
    
    # Load training history if resuming
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        if 'training_history' in checkpoint:
            training_history = checkpoint['training_history']
            print(f"Loaded training history: {len(training_history['losses'])} epochs")

    for epoch in range(start_epoch, start_epoch + args.num_train_epochs):
        bucket_sampler.set_epoch(epoch)
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        epoch_loss_sum = 0.0
        epoch_loss_ce_sum = 0.0
        epoch_token_sum = 0
        epoch_correct_sum = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + args.num_train_epochs}")):
            B = batch["labels"].size(0)
            labels = batch["labels"].to(device)  # (B, T_assistant)
            feat = batch['embeddings'].to(device)  # (B, seq, hidden)

            assistant_start = batch['assistant_start']
            if isinstance(assistant_start, torch.Tensor):
                assistant_start = assistant_start.tolist()
            elif not isinstance(assistant_start, (list, tuple)):
                assistant_start = [int(assistant_start)]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = head(feat, assistant_start)  # (B, max_assist_len, num_labels)
                
                # Ensure labels match logits shape
                if logits.size(1) != labels.size(1):
                    # Adjust labels to match logits length
                    if logits.size(1) < labels.size(1):
                        labels = labels[:, :logits.size(1)]
                    else:
                        # Pad labels if needed
                        pad_len = logits.size(1) - labels.size(1)
                        labels = F.pad(labels, (0, pad_len), value=-100)
                
                loss_ce = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = loss_ce / args.gradient_acc_steps

            loss.backward()

            with torch.no_grad():
                loss_val = loss.item()
                total_loss += loss_val
                preds = logits.argmax(dim=-1)  # (B, T_assistant)
                mask = (labels != -100)
                correct = (preds[mask] == labels[mask]).sum().item()
                num_tokens = mask.sum().item()
                total_correct += correct
                total_tokens += num_tokens
                
                epoch_loss_sum += loss_val * args.gradient_acc_steps
                epoch_loss_ce_sum += loss_ce.item()
                epoch_correct_sum += correct
                epoch_token_sum += num_tokens

            if (step + 1) % args.gradient_acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                completed_steps += 1
                global_step += 1

                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = total_loss / args.gradient_acc_steps
                avg_acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
                print(f"Epoch [{epoch+1}/{start_epoch + args.num_train_epochs}], "
                    f"UpdateStep [{completed_steps}/{total_training_steps}], "
                    f"LR: {current_lr:.2e}, Loss: {avg_loss:.4f}, Acc(token): {avg_acc:.4f}")

                if args.use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/accuracy": avg_acc,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/step": completed_steps
                    })

                total_loss = 0.0
                total_correct = 0
                total_tokens = 0

                if max_steps is not None and max_steps > 0 and completed_steps >= max_steps:
                    break

        if max_steps is not None and max_steps > 0 and completed_steps >= max_steps:
            print("Reached max_steps. Stopping training.")
            break

        # Record epoch metrics and plot
        epoch_avg_loss = epoch_loss_sum / len(train_loader)
        epoch_avg_loss_ce = epoch_loss_ce_sum / len(train_loader)
        epoch_avg_acc = epoch_correct_sum / epoch_token_sum if epoch_token_sum > 0 else 0.0
        
        training_history['losses'].append(epoch_avg_loss)
        training_history['losses_ce'].append(epoch_avg_loss_ce)
        training_history['accs'].append(epoch_avg_acc)
        
        print(f"Epoch {epoch+1} - Loss: {epoch_avg_loss:.4f} (CE: {epoch_avg_loss_ce:.4f}), Acc: {epoch_avg_acc:.4f}")
        
        plot_path = plot_training_curves(training_history['losses'], training_history['accs'], args.save_dir, loss_suffix, model_name_for_file)

        if args.use_wandb:
            wandb.log({
                "epoch/loss": epoch_avg_loss,
                "epoch/loss_ce": epoch_avg_loss_ce,
                "epoch/accuracy": epoch_avg_acc,
                "epoch/number": epoch + 1,
                "epoch/training_curves": wandb.Image(plot_path),
            })

        ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch}_{model_name_for_file}_{loss_suffix}.pt")
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
            'completed_steps': completed_steps,
            'training_history': training_history,
        }
        torch.save(checkpoint_dict, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path} (epoch {epoch})")

        if args.use_wandb and args.wandb_log_artifacts:
            artifact = wandb.Artifact(
                name=f"checkpoint-epoch-{epoch}",
                type="model",
                metadata={"epoch": epoch, "loss": epoch_avg_loss, "accuracy": epoch_avg_acc},
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    print("Training complete!")
    print(f'Final checkpoint: {ckpt_path}')
    
    if args.use_wandb:
        wandb.finish()




def load_config(config_path):
    """Load YAML config file. Returns empty dict if file not found."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    # Pre-parse to get --config path before building the full parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(os.path.dirname(script_dir), "configs", "train.yaml")
    pre_parser.add_argument("--config", type=str, default=default_config)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Train the StreamingHead. Edit configs/train.yaml or pass --key value to override.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to YAML config file")

    # --- Model & Path ---
    parser.add_argument("--efficient_data", type=str, help="Path to efficient (FCS) training data")
    parser.add_argument("--overthinking_data", type=str, help="Path to overthinking training data")
    parser.add_argument("--model_name", type=str, help="HuggingFace model ID or local path")
    parser.add_argument("--save_dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--idx_layer", type=int, help="Transformer layer index for feature extraction")

    # --- Training ---
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gradient_acc_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for LR scheduler")
    parser.add_argument("--deterministic", action="store_true", default=None, help="Deterministic mode (slower)")

    # --- Checkpoint ---
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--max_build_samples", type=int, help="Batch size for cache building (None = all at once)")

    # --- Wandb ---
    parser.add_argument("--use_wandb", action="store_true", default=None, help="Enable wandb logging")
    parser.add_argument("--no_wandb", action="store_true", default=None, help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name (auto-generated if omitted)")
    parser.add_argument("--wandb_log_artifacts", action="store_true", default=None, help="Upload checkpoints as W&B artifacts")

    # Apply YAML defaults, then parse CLI (CLI overrides YAML)
    parser.set_defaults(**config)
    args = parser.parse_args()

    # Handle no_wandb flag
    if args.no_wandb:
        args.use_wandb = False

    setup_hf_cache()
    set_seed(42, deterministic=bool(args.deterministic))
    train(args)


if __name__ == "__main__":
    main()
