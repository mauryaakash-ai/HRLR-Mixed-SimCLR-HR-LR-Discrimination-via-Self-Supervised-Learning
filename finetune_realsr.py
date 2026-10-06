import os
import re
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from resnet import ResNetSimCLR


# Matches both naming conventions seen across the two folders:
#   all_images/   -> canon_hr_0001.png   , canon_lr_0001.png
#   val_dataset/  -> Canon_001_HR.png    , Canon_001_LR4.png
HR_PATTERN = re.compile(r'_hr[_.]')
LR_PATTERN = re.compile(r'_lr\d*[_.]')


def is_hr_file(filename):
    return HR_PATTERN.search(filename.lower()) is not None


def is_lr_file(filename):
    return LR_PATTERN.search(filename.lower()) is not None


class MixedHRLRDataset(Dataset):
    """
    HR and LR images loaded from a single folder (no internal split).
    Handles two filename conventions:
      canon_hr_0001.png / canon_lr_0001.png
      Canon_001_HR.png  / Canon_001_LR4.png
    HR (sharp)     -> label 0
    LR (degraded)  -> label 1
    """
    def __init__(self, data_dir, patch_size=96,
                 patches_per_image=30, split='train'):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

        all_files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith('.png'))
        hr_files = [f for f in all_files if is_hr_file(f)]
        lr_files = [f for f in all_files if is_lr_file(f)]

        print(f"[{split}] Found {len(hr_files)} HR files and {len(lr_files)} LR files in {data_dir}")
        print(f"[{split}] Loading {len(hr_files)} HR + {len(lr_files)} LR images...")
        self.hr_images, self.lr_images = [], []
        for f in hr_files:
            self.hr_images.append(
                Image.open(os.path.join(data_dir, f)).convert('RGB'))
        for f in lr_files:
            self.lr_images.append(
                Image.open(os.path.join(data_dir, f)).convert('RGB'))
        print(f"[{split}] Cached! {len(self.hr_images)} HR + {len(self.lr_images)} LR")

        # Track source (canon/div2k/nikon) per image for per-source eval later
        self.hr_sources = [self._get_source(f) for f in hr_files]
        self.lr_sources = [self._get_source(f) for f in lr_files]

        # Light augmentation only for training -> helps close the train/val gap
        # (only horizontal flip; vertical flip removed — it hurt signal too much)
        if split == 'train':
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        # Precompute patch locations
        self.hr_locations = []
        self.lr_locations = []
        for img_idx, hr_img in enumerate(self.hr_images):
            w, h = hr_img.size
            max_x = max(w - patch_size, 0)
            max_y = max(h - patch_size, 0)
            for _ in range(patches_per_image):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                self.hr_locations.append((img_idx, x, y))

        for img_idx, lr_img in enumerate(self.lr_images):
            w, h = lr_img.size
            max_x = max(w - patch_size, 0)
            max_y = max(h - patch_size, 0)
            for _ in range(patches_per_image):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                self.lr_locations.append((img_idx, x, y))

    @staticmethod
    def _get_source(filename):
        name = filename.lower()
        for src in ('canon', 'div2k', 'nikon'):
            if src in name:
                return src
        return 'unknown'

    def source_of(self, idx):
        """Return the source (canon/div2k/nikon) for a given dataset index."""
        n_hr = len(self.hr_locations)
        if idx < n_hr:
            img_idx, _, _ = self.hr_locations[idx]
            return self.hr_sources[img_idx]
        else:
            img_idx, _, _ = self.lr_locations[idx - n_hr]
            return self.lr_sources[img_idx]

    def __len__(self):
        return len(self.hr_locations) + len(self.lr_locations)

    def __getitem__(self, idx):
        n_hr = len(self.hr_locations)
        ps = self.patch_size
        if idx < n_hr:
            img_idx, x, y = self.hr_locations[idx]
            patch = self.hr_images[img_idx].crop((x, y, x + ps, y + ps))
            return self.transform(patch), 0
        else:
            img_idx, x, y = self.lr_locations[idx - n_hr]
            patch = self.lr_images[img_idx].crop((x, y, x + ps, y + ps))
            return self.transform(patch), 1


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item() * 100


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_dir  = 'Dataset/mixed_dataset/all_images'   # 1000 images: canon+div2k+nikon, HR+LR
    val_dir    = 'Dataset/val_dataset'                # 200 images: separate held-out val set
    checkpoint = './checkpoints_mixed/checkpoint_epoch0100.pth'
    save_dir   = './checkpoints_mixed'
    patch_size        = 96
    patches_per_image = 30
    batch_size        = 64
    epochs            = 150
    num_workers       = 1

    os.makedirs(save_dir, exist_ok=True)

    train_ds = MixedHRLRDataset(train_dir, patch_size, patches_per_image, split='train')
    val_ds   = MixedHRLRDataset(val_dir, patch_size, patches_per_image, split='val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Load pretrained encoder — UNFROZEN for fine-tuning
    model = ResNetSimCLR(depth=18, proj_out_dim=128)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    encoder = model.backbone.to(device)
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad = True

    classifier = nn.Sequential(
        nn.Dropout(p=0.15),
        nn.Linear(512, 2)
    ).to(device)
    nn.init.normal_(classifier[1].weight, std=0.01)
    nn.init.zeros_(classifier[1].bias)

    # Two LR groups — small for encoder, larger for head.
    # Tuned down from the previous over-regularized run (encoder LR raised
    # back up a bit, weight_decay lowered) so the model can actually fit.
    optimizer = torch.optim.SGD([
        {'params': encoder.parameters(),    'lr': 0.0008},
        {'params': classifier.parameters(), 'lr': 0.01}
    ], momentum=0.9, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\nStarting fine-tuning on mixed dataset ({epochs} epochs)...")
    best_acc = 0.0
    patience = 8           # stop if val acc doesn't improve for this many epochs
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        total_loss, total_acc = 0.0, 0.0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            feats  = encoder(imgs)
            logits = classifier(feats)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients -> prevents sudden large updates that were
            # likely causing the epoch-11 val accuracy spike/drop.
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_acc  += accuracy(logits, labels)
        scheduler.step()

        encoder.eval()
        classifier.eval()
        val_acc = 0.0
        source_correct = {}
        source_total = {}
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                feats  = encoder(imgs)
                logits = classifier(feats)
                val_acc += accuracy(logits, labels)

                # Per-source breakdown (only computed on the final epoch to save time)
                if epoch == epochs:
                    preds = logits.argmax(dim=1)
                    start = batch_idx * val_loader.batch_size
                    for i in range(len(labels)):
                        src = val_ds.source_of(start + i)
                        source_total[src] = source_total.get(src, 0) + 1
                        if preds[i].item() == labels[i].item():
                            source_correct[src] = source_correct.get(src, 0) + 1

        train_acc = total_acc / len(train_loader)
        val_acc   = val_acc   / len(val_loader)
        print(f"Epoch [{epoch:2d}/{epochs}] "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'encoder':    encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, os.path.join(save_dir, 'best_finetune_mixedA25.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping: no val improvement for {patience} epochs.")
                break

    print(f"\n=== Best Val Accuracy: {best_acc:.2f}% ===")
    print("HR=0 (clean), LR=1 (degraded) — Mixed dataset (canon, div2k, nikon)")

    if source_total:
        print("\nPer-source accuracy (last epoch):")
        for src in sorted(source_total.keys()):
            acc = 100.0 * source_correct.get(src, 0) / source_total[src]
            print(f"  {src:8s}: {acc:.2f}%  ({source_correct.get(src, 0)}/{source_total[src]})")


if __name__ == '__main__':
    main()