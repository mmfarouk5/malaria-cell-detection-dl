import os
import yaml
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from data import train_loader, val_loader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


logger = setup_logger()

def load_config(path="../config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_resnet50(num_classes=1, freeze_backbone=False):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Training"):
        imgs = imgs.to(device)
        labels = labels.float().to(device)

        try:
            optimizer.zero_grad()
            outputs = model(imgs).squeeze(1)

            loss = criterion(outputs, labels)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.detach().item()

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        except RuntimeError as e:
            logger.error(f"Training error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    return total_loss / len(loader), accuracy


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    return val_loss / len(loader), accuracy


def main():

    cfg = load_config()
    logger.info("Loaded configuration.")

    writer = SummaryWriter(log_dir=cfg["paths"]["log_dir"])
    logger.info("TensorBoard writer initialized.")

    model = build_resnet50(
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    save_path = cfg["paths"]["save_dir"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = cfg["training"]["epochs"]
    best_val_loss = float('inf')
    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")

    logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    main()