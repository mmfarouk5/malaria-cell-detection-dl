from torch.utils.data import DataLoader
from torchvision import datasets, transforms


train_dir = "../data/preprocessed_data/train"
val_dir = "../data/preprocessed_data/val"
test_dir = "../data/preprocessed_data/test"

BATCH_SIZE = 32

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

val_test_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(root=val_dir, transform=val_test_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

if __name__ == "__main__":
    print(f"Train images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Test images: {len(test_data)}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")