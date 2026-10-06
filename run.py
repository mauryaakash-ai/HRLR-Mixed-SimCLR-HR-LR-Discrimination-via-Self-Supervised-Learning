import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from resnet import ResNetSimCLR
from objective import NTXentLoss
from model_util import LARS, get_lr_schedule
from data import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR on DIV2K HR patches')
    parser.add_argument('--hr_dir', type=str,
                        default='Dataset/DIV2K_train_HR')
    parser.add_argument('--model_dir', type=str, default='./checkpoints')
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--patches_per_image', type=int, default=50)
    parser.add_argument('--resnet_depth', type=int, default=18)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--proj_out_dim', type=int, default=128)
    parser.add_argument('--num_proj_layers', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--checkpoint_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def save_checkpoint(state, model_dir, epoch):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'checkpoint_epoch{epoch:04d}.pth')
    torch.save(state, path)
    torch.save(state, os.path.join(model_dir, 'checkpoint_latest.pth'))
    print(f"Checkpoint saved: {path}")


def train_one_epoch(model, loader, criterion, optimizer,
                    scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for step, views in enumerate(pbar):
        # views: (batch, 2, C, H, W)
        views = views.transpose(0, 1).reshape(-1, *views.shape[2:]).to(device)

        optimizer.zero_grad()
        _, z = model(views)
        loss, logits_ab, labels = criterion(z)
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = (logits_ab.argmax(dim=1) == labels).float().mean().item()
        total_loss += loss.item()
        total_acc += acc
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    n = len(loader)
    return total_loss / n, total_acc / n


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = build_dataloaders(
        hr_dir=args.hr_dir,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers
    )
    print(f"Dataset: {len(train_loader)} batches/epoch")

    model = ResNetSimCLR(
        depth=args.resnet_depth,
        proj_out_dim=args.proj_out_dim,
        num_proj_layers=args.num_proj_layers
    ).to(device)
    print(f"Model: ResNet-{args.resnet_depth} + {args.num_proj_layers}-layer projection head")

    criterion = NTXentLoss(temperature=args.temperature, hidden_norm=True)

    scaled_lr = args.learning_rate * args.train_batch_size / 256.0
    optimizer = LARS(
        model.parameters(),
        lr=scaled_lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    total_steps = args.train_epochs * len(train_loader)
    warmup_steps = min(int(args.warmup_epochs * len(train_loader)),
                       total_steps - 1)
    scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps, scaled_lr)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    print(f"\nStarting pretraining for {args.train_epochs} epochs...")
    print(f"scaled_lr={scaled_lr:.6f} | warmup={warmup_steps} steps | total={total_steps} steps\n")

    for epoch in range(start_epoch, args.train_epochs + 1):
        loss, acc = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, device, epoch
        )
        print(f"Epoch [{epoch:3d}/{args.train_epochs}] "
              f"Loss: {loss:.4f} | Contrast Acc: {acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % args.checkpoint_epochs == 0:
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }, args.model_dir, epoch)

    print("Pretraining complete!")
    save_checkpoint({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': args.train_epochs,
        'loss': loss,
    }, args.model_dir, args.train_epochs)


if __name__ == '__main__':
    main()
