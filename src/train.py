import argparse
import time
from pathlib import Path

import yaml
import torch

from model import LightweightFaceDetector
from loss import LightweightFaceLoss
from dataset import get_dataloader
from transforms import build_transforms


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="configs/base.yml",
        help="Path to YAML config file."
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--run_name", type=str, default=None)

    # TODO: --use_wandb

    args = parser.parse_args()
    return args


def load_config(config_path, args):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['loader']['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['train']['lr'] = args.lr
    if args.device is not None:
        cfg['train']['device'] = args.device
    if args.run_name is not None:
        cfg['train']['run_name'] = args.run_name

    return cfg


def collate_fn(batch):
    # TODO
    raise NotImplementedError


def validate(model, val_loader, criterion, device):
    model.eval()
    total_losses = {}
    num_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # targets = {k: v.to(device) for k, v in targets.items()}

            y_pred = model(images)
            losses = criterion(y_pred, targets)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            num_batches += 1

    model.train()
    return {k: v / num_batches for k, v in total_losses.items()}


def train(cfg):
    device = torch.device(cfg['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # --- Data ---
    train_transform = build_transforms(cfg, split="train")
    val_transform = build_transforms(cfg, split="val")

    train_loader = get_dataloader(cfg['loader'], split="train", transform=train_transform)
    val_loader = get_dataloader(cfg['loader'], split="val", transform=val_transform)

    # --- Model / Loss / Optimizer ---
    model = LightweightFaceDetector(pretrained=cfg['model'].get('pretrained', True)).to(device)
    criterion = LightweightFaceLoss(
        w_cls=cfg['loss'].get('w_cls', 1.0),
        w_box=cfg['loss'].get('w_box', 1.0),
        w_kps=cfg['loss'].get('w_kps', 0.5),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])

    # TODO: scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])

    start_epoch = 0
    if cfg['train'].get('resume'):
        checkpoint = torch.load(cfg['train']['resume'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # --- Logging setup ---
    run_name = cfg['train'].get('run_name', f"run_{int(time.time())}")
    log_dir = Path(cfg['train'].get('log_dir', 'logs')) / run_name
    ckpt_dir = Path(cfg['train'].get('ckpt_dir', 'checkpoints')) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TODO: wandb.init(project=..., name=run_name, config=cfg) if --use_wandb

    val_every = cfg['train'].get('val_every_epochs', 1)
    log_every = cfg['train'].get('log_every_steps', 50)

    # --- Training loop ---
    model.train()
    for epoch in range(start_epoch, cfg['train']['epochs']):
        epoch_start = time.time()
        running_losses = {}

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            # targets = {k: v.to(device) for k, v in targets.items()}  # ajustar según estructura final

            optimizer.zero_grad()
            y_pred = model(images)
            losses = criterion(y_pred, targets)

            losses['total_loss'].backward()
            optimizer.step()

            for k, v in losses.items():
                running_losses[k] = running_losses.get(k, 0.0) + v.item()

            if (step + 1) % log_every == 0:
                avg = {k: v / log_every for k, v in running_losses.items()}
                print(f"[epoch {epoch}] step {step+1}/{len(train_loader)} "
                      f"loss={avg['total_loss']:.4f} "
                      f"cls={avg['loss_cls']:.4f} box={avg['loss_box']:.4f} kps={avg['loss_kps']:.4f}")
                # TODO: wandb.log({f"train/{k}": v for k, v in avg.items()}, step=global_step)
                running_losses = {}

        # if scheduler is not None:
        #     scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"[epoch {epoch}] done in {epoch_time:.1f}s")

        # --- Validación periódica ---
        if (epoch + 1) % val_every == 0:
            val_losses = validate(model, val_loader, criterion, device)
            print(f"[epoch {epoch}] val_loss={val_losses['total_loss']:.4f}")
            # TODO: wandb.log({f"val/{k}": v for k, v in val_losses.items()}, step=global_step)

        # --- Checkpoint ---
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': cfg,
        }, ckpt_path)


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.config, args)
    train(cfg)
