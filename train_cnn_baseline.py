import argparse, json, os, random, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(pred_logits, y):
    preds = pred_logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

def get_optimizer(name, params, lr, weight_decay, momentum):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=0.0, use_bn=True):
        super().__init__()
        pad = k // 2  # "same" padding for odd kernels
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(inplace=True)]
        if p > 0:
            layers.append(nn.Dropout2d(p))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ModularCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, depth=3, base_filters=32, kernel_size=3, dropout=0.1, use_bn=True):
        super().__init__()
        channels = [in_channels] + [base_filters * (2 ** i) for i in range(depth)]
        blocks = []
        for i in range(depth):
            blocks.append(
                ConvBlock(channels[i], channels[i+1], k=kernel_size, p=dropout, use_bn=use_bn)
            )
        self.features = nn.Sequential(*blocks)
        # spatially collapse to 1x1 regardless of input size
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

def build_transforms(dataset):
    if dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return t, t
    elif dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return t_train, t_test
    else:
        raise ValueError("dataset must be 'mnist' or 'cifar10'")

def load_dataset(root, dataset, val_ratio=0.1, seed=42):
    t_train, t_test = build_transforms(dataset)
    if dataset == "mnist":
        train_full = datasets.MNIST(root, train=True, download=True, transform=t_train)
        test = datasets.MNIST(root, train=False, download=True, transform=t_test)
        in_ch, num_classes = 1, 10
    else:  # cifar10
        train_full = datasets.CIFAR10(root, train=True, download=True, transform=t_train)
        test = datasets.CIFAR10(root, train=False, download=True, transform=t_test)
        in_ch, num_classes = 3, 10

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    # make split reproducible
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(train_full, [train_size, val_size], generator=g)
    return train, val, test, in_ch, num_classes

def run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss = 0.0
    epoch_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y) if criterion else F.cross_entropy(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()
        bsz = x.size(0)
        epoch_loss += loss.item() * bsz
        epoch_acc += accuracy(logits.detach(), y) * bsz
        n += bsz
    return epoch_loss / n, epoch_acc / n


def main():
    parser = argparse.ArgumentParser(description="Modular CNN Baseline (MNIST/CIFAR10)")
    # knobs
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_bn", action="store_true")

    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./results")

    args = parser.parse_args()

    # seeds and device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_set, val_set, test_set, in_ch, num_classes = load_dataset(
        args.data_dir, args.dataset, val_ratio=args.val_ratio, seed=args.seed
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # model
    model = ModularCNN(
        in_channels=in_ch,
        num_classes=num_classes,
        depth=args.depth,
        base_filters=args.base_filters,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_bn=args.use_bn
    ).to(device)

    # optimizer / loss
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum)
    criterion = nn.CrossEntropyLoss()

    # results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_d{args.depth}_f{args.base_filters}_k{args.kernel_size}_do{args.dropout}_{args.optimizer}_lr{args.lr}_{timestamp}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config
    config = vars(args).copy()
    config.update({"in_channels": in_ch, "num_classes": num_classes, "device": str(device), "run_name": run_name})
    save_json(config, out_dir / "config.json")

    # training
    best_val_acc = 0.0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer=optimizer, criterion=criterion)
        va_loss, va_acc = run_epoch(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc
        })

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"| val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        # checkpoint
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

        # save rolling history each epoch
        save_json(history, out_dir / "metrics.json")

    train_seconds = time.time() - start_time

    # load best and test
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    te_loss, te_acc = run_epoch(model, test_loader, device)

    summary = {
        "best_val_acc": best_val_acc,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "train_seconds": train_seconds,
        "params": sum(p.numel() for p in model.parameters())
    }
    print(f"\n== Summary ==\n"
          f"best_val_acc: {best_val_acc:.4f}\n"
          f"test_acc:     {te_acc:.4f}\n"
          f"params:       {summary['params']}\n"
          f"train_time:   {train_seconds:.1f}s")

    save_json(summary, out_dir / "summary.json")

if __name__ == "__main__":
    main()
