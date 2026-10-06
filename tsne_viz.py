import os
import random
import torch
import numpy as np
from torchvision import transforms as T
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from resnet import ResNetSimCLR


def extract_features(encoder, val_dir, patch_size=96,
                     patches_per_image=5, device='cuda'):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    hr_files = sorted([f for f in os.listdir(val_dir) if '_HR.png' in f])
    lr_files = sorted([f for f in os.listdir(val_dir) if '_LR4.png' in f])

    features, labels, sources = [], [], []
    encoder.eval()

    with torch.no_grad():
        for f in hr_files:
            src = 'Canon' if 'Canon' in f else 'Nikon'
            img = Image.open(os.path.join(val_dir, f)).convert('RGB')
            w, h = img.size
            for _ in range(patches_per_image):
                x = random.randint(0, max(w-patch_size, 0))
                y = random.randint(0, max(h-patch_size, 0))
                patch = img.crop((x, y, x+patch_size, y+patch_size))
                t = transform(patch).unsqueeze(0).to(device)
                feat = encoder(t).squeeze().cpu().numpy()
                features.append(feat)
                labels.append(0)  # HR
                sources.append(src)

        for f in lr_files:
            src = 'Canon' if 'Canon' in f else 'Nikon'
            img = Image.open(os.path.join(val_dir, f)).convert('RGB')
            w, h = img.size
            for _ in range(patches_per_image):
                x = random.randint(0, max(w-patch_size, 0))
                y = random.randint(0, max(h-patch_size, 0))
                patch = img.crop((x, y, x+patch_size, y+patch_size))
                t = transform(patch).unsqueeze(0).to(device)
                feat = encoder(t).squeeze().cpu().numpy()
                features.append(feat)
                labels.append(1)  # LR
                sources.append(src)

    return np.array(features), np.array(labels), sources


def plot_tsne(X_2d, labels, sources, title, ax):
    colors = {(0, 'Canon'): '#2196F3', (0, 'Nikon'): '#03A9F4',
              (1, 'Canon'): '#F44336', (1, 'Nikon'): '#FF5722'}
    legend_items = {
        'HR Canon': '#2196F3', 'HR Nikon': '#03A9F4',
        'LR Canon': '#F44336', 'LR Nikon': '#FF5722'
    }
    for i in range(len(X_2d)):
        key = (labels[i], sources[i])
        ax.scatter(X_2d[i, 0], X_2d[i, 1],
                   c=colors[key], alpha=0.6, s=15)

    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=l) for l, c in legend_items.items()]
    ax.legend(handles=legend, loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.grid(True, alpha=0.3)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    val_dir    = 'Dataset/val_dataset'
    checkpoint = './checkpoints_mixed/checkpoint_epoch0100.pth'

    # Load pretrained encoder
    model = ResNetSimCLR(depth=18, proj_out_dim=128)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    encoder_pretrained = model.backbone.to(device)
    encoder_pretrained.eval()
    for p in encoder_pretrained.parameters():
        p.requires_grad = False

    # Random encoder
    model_rand = ResNetSimCLR(depth=18, proj_out_dim=128)
    encoder_random = model_rand.backbone.to(device)
    encoder_random.eval()

    print("Extracting features from pretrained encoder...")
    X_pre, y, sources = extract_features(encoder_pretrained, val_dir)
    print(f"Features: {X_pre.shape}")

    print("Extracting features from random encoder...")
    X_rand, _, _ = extract_features(encoder_random, val_dir)

    # t-SNE
    print("Running t-SNE on pretrained features...")
    scaler = StandardScaler()
    X_pre_s  = scaler.fit_transform(X_pre)
    X_rand_s = StandardScaler().fit_transform(X_rand)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_pre_2d  = tsne.fit_transform(X_pre_s)

    print("Running t-SNE on random features...")
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_rand_2d = tsne2.fit_transform(X_rand_s)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('SimCLR Feature Space: HR vs LR (RealSR Dataset)',
                 fontsize=13, fontweight='bold')

    plot_tsne(X_rand_2d, y, sources,
              'Random (Untrained) Encoder', axes[0])
    plot_tsne(X_pre_2d,  y, sources,
              'SimCLR Pretrained Encoder\n(Mixed HR+LR, 100 epochs)', axes[1])

    plt.tight_layout()
    plt.savefig('tsne_hr_lr_comparison.png', dpi=150, bbox_inches='tight')
    print("\nt-SNE saved: tsne_hr_lr_comparison.png")

    # Stats
    hr_mask = (y == 0)
    lr_mask = (y == 1)
    print(f"\nFeature space separation (pretrained):")
    hr_center = X_pre_2d[hr_mask].mean(axis=0)
    lr_center = X_pre_2d[lr_mask].mean(axis=0)
    dist = np.linalg.norm(hr_center - lr_center)
    print(f"  HR centroid: {hr_center}")
    print(f"  LR centroid: {lr_center}")
    print(f"  Centroid distance: {dist:.2f}")

    print(f"\nFeature space separation (random):")
    hr_center_r = X_rand_2d[hr_mask].mean(axis=0)
    lr_center_r = X_rand_2d[lr_mask].mean(axis=0)
    dist_r = np.linalg.norm(hr_center_r - lr_center_r)
    print(f"  Centroid distance: {dist_r:.2f}")


if __name__ == '__main__':
    main()
