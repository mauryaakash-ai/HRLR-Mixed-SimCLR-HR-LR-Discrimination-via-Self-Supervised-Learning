import os
import re
import random
import torch
import numpy as np
from torchvision import transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from resnet import ResNetSimCLR


# Matches both naming conventions:
#   canon_hr_0001.png / canon_lr_0001.png   (all_images convention)
#   Canon_001_HR.png  / Canon_001_LR4.png   (val_dataset / RealSR convention)
HR_PATTERN = re.compile(r'_hr[_.]')
LR_PATTERN = re.compile(r'_lr\d*[_.]')


def is_hr_file(filename):
    return HR_PATTERN.search(filename.lower()) is not None


def is_lr_file(filename):
    return LR_PATTERN.search(filename.lower()) is not None


def get_pair_id(filename):
    """
    Strip the hr/lr token to get an id that matches an HR file with its
    corresponding LR file, e.g.:
      Canon_001_HR.png / Canon_001_LR4.png -> canon_001
      div2k_hr_0801.png / div2k_lr_0801.png -> div2k_0801
    """
    name = os.path.splitext(filename)[0].lower()
    name = re.sub(r'_lr\d*', '', name)
    name = re.sub(r'_hr', '', name)
    return name


def get_source(filename):
    name = filename.lower()
    for src in ('canon', 'div2k', 'nikon'):
        if src in name:
            return src
    return 'unknown'


def extract_features(encoder, val_dir, patch_size=96,
                     patches_per_image=10, device='cuda'):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    all_files = sorted(f for f in os.listdir(val_dir) if f.lower().endswith('.png'))
    hr_files = [f for f in all_files if is_hr_file(f)]
    lr_files = [f for f in all_files if is_lr_file(f)]

    print(f"Found {len(hr_files)} HR + {len(lr_files)} LR images")

    # Map pair-id -> HR filename, so each LR image can be resized to match
    # its paired HR image's resolution. This matters for sources like DIV2K
    # where the LR file is a raw 4x-smaller bicubic download (unlike
    # Canon/Nikon RealSR pairs, which are already the same size). Without
    # this, the classifier could learn a trivial "image is small = LR" cue
    # for DIV2K instead of a real sharpness/blur cue, which doesn't
    # generalize to unseen DIV2K images.
    hr_size_by_id = {}
    for f in hr_files:
        img = Image.open(os.path.join(val_dir, f)).convert('RGB')
        hr_size_by_id[get_pair_id(f)] = img.size
        img.close()

    features, labels, sources = [], [], []
    encoder.eval()

    with torch.no_grad():
        # HR patches — label 0
        for f in hr_files:
            img = Image.open(os.path.join(val_dir, f)).convert('RGB')
            w, h = img.size
            max_x = max(w - patch_size, 0)
            max_y = max(h - patch_size, 0)
            for _ in range(patches_per_image):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                t = transform(patch).unsqueeze(0).to(device)
                feat = encoder(t).squeeze().cpu().numpy()
                features.append(feat)
                labels.append(0)
                sources.append(get_source(f))

        # LR patches — label 1
        for f in lr_files:
            img = Image.open(os.path.join(val_dir, f)).convert('RGB')

            # Resize LR to match its paired HR resolution if they differ
            # (e.g. DIV2K: LR is a raw 4x-smaller bicubic download).
            pair_id = get_pair_id(f)
            target_size = hr_size_by_id.get(pair_id)
            if target_size is not None and img.size != target_size:
                img = img.resize(target_size, Image.BICUBIC)

            w, h = img.size
            max_x = max(w - patch_size, 0)
            max_y = max(h - patch_size, 0)
            for _ in range(patches_per_image):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                t = transform(patch).unsqueeze(0).to(device)
                feat = encoder(t).squeeze().cpu().numpy()
                features.append(feat)
                labels.append(1)
                sources.append(get_source(f))

    return np.array(features), np.array(labels), np.array(sources)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    val_dir    = 'Dataset/val_dataset'
    checkpoint = './checkpoints_mixed/checkpoint_epoch0100.pth'
    patch_size = 96
    patches_per_image = 10

    # Load pretrained encoder — FROZEN
    model = ResNetSimCLR(depth=18, proj_out_dim=128)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    encoder = model.backbone.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Encoder frozen — checkpoint: {checkpoint}")

    # Extract features
    print(f"\nExtracting features ({patches_per_image} patches/image)...")
    X, y, src = extract_features(encoder, val_dir, patch_size,
                                 patches_per_image, device)
    print(f"Features shape: {X.shape} | Labels: {y.shape}")
    print(f"HR samples: {(y==0).sum()} | LR samples: {(y==1).sum()}")

    # Train/val split — 80/20
    n = len(X)
    idx = np.random.permutation(n)
    train_idx = idx[:int(0.8*n)]
    val_idx   = idx[int(0.8*n):]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    src_val = src[val_idx]

    # Normalize
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # Also test random encoder for comparison
    print(f"\nTesting RANDOM encoder for comparison...")
    model_rand = ResNetSimCLR(depth=18, proj_out_dim=128)
    enc_rand   = model_rand.backbone.to(device)
    enc_rand.eval()
    X_rand, y_rand, _ = extract_features(enc_rand, val_dir,
                                         patch_size, patches_per_image, device)
    X_rand_s = StandardScaler().fit_transform(X_rand)
    clf_rand = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_rand.fit(X_rand_s, y_rand)
    rand_acc = accuracy_score(y_rand, clf_rand.predict(X_rand_s)) * 100
    print(f"Random encoder accuracy: {rand_acc:.2f}%")

    # Train linear classifier on pretrained features
    print(f"\nTraining Logistic Regression on pretrained features...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train)) * 100
    val_acc   = accuracy_score(y_val,   clf.predict(X_val))   * 100

    print(f"\n=== Linear Evaluation Results (RealSR Val Set) ===")
    print(f"Random encoder accuracy:    {rand_acc:.2f}%")
    print(f"Pretrained encoder train:   {train_acc:.2f}%")
    print(f"Pretrained encoder val:     {val_acc:.2f}%")
    print(f"Improvement over random:    +{val_acc - rand_acc:.2f}%")
    print(f"\nClassification Report (Val):")
    print(classification_report(y_val, clf.predict(X_val),
                                target_names=['HR (clean)', 'LR (degraded)']))

    # Per-source breakdown
    print(f"\nPer-source accuracy (val split):")
    val_preds = clf.predict(X_val)
    for s in sorted(set(src_val)):
        mask = src_val == s
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_val[mask], val_preds[mask]) * 100
        print(f"  {s:8s}: {acc:.2f}%  (n={mask.sum()})")


if __name__ == '__main__':
    main()