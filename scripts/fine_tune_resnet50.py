import os
import yaml
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch import autocast
from torch.amp import GradScaler
from data import train_loader, val_loader


def validate_dataloader(loader, loader_name="DataLoader"):
    try:
        if loader is None:
            raise ValueError(f"{loader_name} is None")

        if len(loader) == 0:
            raise ValueError(f"{loader_name} is empty (no batches available)")

        try:
            first_batch = next(iter(loader))
            if not isinstance(first_batch, (tuple, list)) or len(first_batch) != 2:
                raise ValueError(f"{loader_name} batches should return (images, labels), got {type(first_batch)}")

            imgs, labels = first_batch

            if not isinstance(imgs, torch.Tensor):
                raise ValueError(f"{loader_name} images should be torch.Tensor, got {type(imgs)}")

            if not isinstance(labels, torch.Tensor):
                raise ValueError(f"{loader_name} labels should be torch.Tensor, got {type(labels)}")

            if imgs.size(0) == 0:
                raise ValueError(f"{loader_name} has empty batch (batch size is 0)")

            logger.info(f"{loader_name} validation passed: {len(loader)} batches, "
                       f"batch shape {imgs.shape}, labels shape {labels.shape}")
            return True

        except StopIteration:
            raise ValueError(f"{loader_name} iteration failed immediately (empty iterator)")
        except Exception as e:
            raise RuntimeError(f"{loader_name} failed during first batch test: {e}")

    except Exception as e:
        logger.error(f"{loader_name} validation failed: {e}")
        raise


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()


def set_seed(seed: int):
    import random
    import numpy as np

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def load_config(path):
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {path}")
            
            required_keys = ["training", "model", "paths"]
            for key in required_keys:
                if key not in cfg:
                    raise ValueError(f"Missing required config key: {key}")
            
            return cfg
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def build_resnet50(num_classes=1, freeze_backbone=False):
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        logger.info(f"ResNet50 built with {num_classes} output classes; freeze_backbone={freeze_backbone}")
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise


def save_checkpoint(full_path, epoch, model, optimizer, scheduler, best_val_loss, scaler=None):
    try:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        torch.save(checkpoint, full_path)
        logger.info(f"Checkpoint saved: {full_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {full_path}: {e}")


def try_load_checkpoint(path, model, optimizer, scheduler, scaler=None):
    if not path or not os.path.exists(path):
        logger.info("No checkpoint found. Starting from scratch.")
        return 0, float("inf")

    try:
        ckpt = torch.load(path, map_location=device)

        required = ["model_state_dict", "optimizer_state_dict", "epoch", "best_val_loss"]
        if not all(k in ckpt for k in required):
            logger.error("Checkpoint missing required keys. Ignoring.")
            return 0, float("inf")

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Move tensors to device BEFORE loading scheduler
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Couldn't load scheduler state: {e}")


        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                logger.warning(f"Couldn't load scaler state: {e}")

        epoch = ckpt["epoch"]
        best_val_loss = ckpt["best_val_loss"]
        logger.info(f"Resumed from checkpoint {path} at epoch {epoch} with best_val_loss={best_val_loss:.4f}")
        return epoch, best_val_loss

    except Exception as e:
        logger.error(f"Error loading checkpoint {path}: {e}")
        return 0, float("inf")


def train_one_epoch(model, loader, criterion, optimizer, scaler, max_norm, threshold, amp_enabled):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    try:
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Training", leave=True)):
            try:
                imgs = imgs.to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                with autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(imgs).squeeze(1)
                    loss = criterion(outputs, labels)

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    optimizer.step()

                total_loss += loss.item()

                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > threshold).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            except RuntimeError as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                if "out of memory" in str(e).lower():
                    logger.error("GPU out of memory. Try reducing batch size.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise
            except Exception as e:
                logger.error(f"Unexpected error in batch {batch_idx}: {e}")
                raise

    except Exception as e:
        logger.error(f"Training epoch failed: {e}")
        raise

    if total == 0:
        logger.warning("No samples processed during training epoch")
        return 0.0, 0.0

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


def validate(model, loader, criterion, threshold, amp_enabled):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Validation", leave=True)):
                try:
                    imgs = imgs.to(device)
                    labels = labels.float().to(device)

                    with autocast(device_type=device.type, enabled=amp_enabled):
                        outputs = model(imgs).squeeze(1)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    preds = (torch.sigmoid(outputs) > threshold).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                except RuntimeError as e:
                    logger.error(f"Error processing validation batch {batch_idx}: {e}")
                    if "out of memory" in str(e).lower():
                        logger.error("GPU out of memory during validation. Try reducing batch size.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in validation batch {batch_idx}: {e}")
                    raise

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

    if total == 0:
        logger.warning("No samples processed during validation")
        return 0.0, 0.0

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = val_loss / len(loader)
    return avg_loss, accuracy


def main():
    writer = None
    try:
        cfg = load_config("../configs/fine_tune_resnet50.yaml")
        logger.info(f"Device: {device}")

        # Validate dataloaders with comprehensive error checking
        try:
            validate_dataloader(train_loader, "train_loader")
            validate_dataloader(val_loader, "val_loader")
        except (ValueError, RuntimeError) as e:
            logger.error(f"Dataloader validation failed: {e}")
            raise

        set_seed(cfg["training"].get("random_seed", 42))
        logger.info(f"Random seed set to {cfg['training'].get('random_seed', 42)}")

        writer = SummaryWriter(log_dir=cfg["paths"]["log_dir"])
        logger.info(f"TensorBoard writer initialized at {cfg['paths']['log_dir']}")

        model = build_resnet50(
            num_classes=cfg["model"].get("num_classes", 1),
            freeze_backbone=cfg["model"].get("freeze_backbone", False)
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["training"].get("learning_rate", 1e-4),
            weight_decay=cfg["training"].get("weight_decay", 1e-4)
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg["training"].get("scheduler_factor", 0.5),
            patience=cfg["training"].get("scheduler_patience", 3)
        )

        amp_enabled = cfg.get("amp", {}).get("enabled", True)
        scaler = GradScaler(device=device.type, enabled=amp_enabled)
        logger.info(f"AMP enabled: {amp_enabled} on device: {device.type}")

        save_path = cfg["paths"]["save_dir"]
        checkpoint_main = cfg["paths"]["checkpoint_dir"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_main), exist_ok=True)

        epochs = cfg["training"].get("epochs", 12)
        max_norm = cfg["training"].get("max_norm", 1.0)
        threshold = cfg["training"].get("threshold", 0.5)
        early_stopping_patience = cfg["training"].get("early_stopping_patience", 5)
        resume_training = cfg["training"].get("resume_training", False)

        start_epoch = 0
        best_val_loss = float("inf")

        if resume_training:
            start_epoch, best_val_loss = try_load_checkpoint(checkpoint_main, model, optimizer, scheduler, scaler)
            if start_epoch > 0:
                logger.info("Performing quick validation check of loaded checkpoint...")
                val_loss, val_acc = validate(model, val_loader, criterion, threshold, amp_enabled)
                logger.info(f"Checkpoint validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        patience_counter = 0
        logger.info(f"Starting training for {epochs} epochs (from epoch {start_epoch + 1})...")

        for epoch in range(start_epoch + 1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, max_norm, threshold, amp_enabled)
            val_loss, val_acc = validate(model, val_loader, criterion, threshold, amp_enabled)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                logger.info(f"Best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

            save_checkpoint(checkpoint_main, epoch, model, optimizer, scheduler, best_val_loss, scaler=scaler)

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after epoch {epoch}")
                break

        logger.info(f"Training finished. Best validation loss: {best_val_loss:.4f}")

        writer.close()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if writer is not None:
            writer.close()
            logger.info("TensorBoard writer closed")


if __name__ == "__main__":
    main()
