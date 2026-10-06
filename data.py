import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, ImageFilter


class GaussianBlur:
    def __init__(self, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class SimCLRTransform:
    """Two random augmented views — same as official SimCLR augmentation."""
    def __init__(self, patch_size=96, color_jitter_strength=1.0):
        s = color_jitter_strength
        self.transform = T.Compose([
            T.RandomResizedCrop(patch_size, scale=(0.2, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(0.1, 2.0)], p=0.0),  # OFF — preserve sharpness features
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class DIV2KPatchDataset(Dataset):
    """
    Extracts random patches from DIV2K HR images on-the-fly.
    Each __getitem__ returns two augmented views of a random patch.
    """
    def __init__(self, hr_dir, patch_size=96,
                 patches_per_image=50, transform=None):
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform

        files = sorted([
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"Found {len(files)} HR images — loading into memory...")
        self.images = []
        for i, fpath in enumerate(files):
            img = Image.open(fpath).convert('RGB')
            self.images.append(img)
            if (i+1) % 100 == 0:
                print(f"  Loaded {i+1}/{len(files)}")
        print(f"All images cached! Virtual dataset size: {len(self.images) * patches_per_image} patches")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        # Which image and which patch within it
        img_idx = idx // self.patches_per_image
        img = self.images[img_idx]
        w, h = img.size

        # Random crop a patch (at least patch_size x patch_size)
        max_x = w - self.patch_size
        max_y = h - self.patch_size
        if max_x <= 0 or max_y <= 0:
            img = img.resize((self.patch_size * 2, self.patch_size * 2),
                             Image.BICUBIC)
            max_x = max_y = self.patch_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        if self.transform:
            view1, view2 = self.transform(patch)
            return torch.stack([view1, view2], dim=0)
        return patch


def build_dataloaders(hr_dir, patch_size=96, patches_per_image=50,
                      batch_size=256, num_workers=1):
    transform = SimCLRTransform(patch_size=patch_size)
    dataset = DIV2KPatchDataset(
        hr_dir=hr_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    return loader


if __name__ == '__main__':
    hr_dir = 'Dataset/mixed_dataset/all_images'
    loader = build_dataloaders(hr_dir, patch_size=96,
                               patches_per_image=50, batch_size=4)
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")   # (4, 2, 3, 96, 96)
    print(f"Total batches: {len(loader)}")
    print("data.py OK!")
