import argparse, json, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=0.0, use_bn=True):
        super().__init__()
        pad = k // 2
        layers = [nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if p > 0:
            layers.append(nn.Dropout2d(p))
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class FewShotCNN(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, depth=4, kernel_size=3, dropout=0.0, use_bn=True, n_way=5):
        super().__init__()
        chs = [in_channels] + [base_filters] * depth
        blocks = []
        for i in range(depth):
            blocks.append(ConvBlock(chs[i], chs[i+1], k=kernel_size, p=dropout, use_bn=use_bn))
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_filters, n_way)
        )

    def forward(self, x):
        return self.head(self.features(x))


class ClassIndexedDataset(Dataset):
    def __init__(self, base, num_classes):
        self.base = base
        self.num_classes = num_classes
        self.by_class: List[List[int]] = [[] for _ in range(num_classes)]
        for idx, (_, y) in enumerate(base):
            self.by_class[y].append(idx)

    def sample_episode(self, n_way: int, k_shot: int, q_query: int, rng: np.random.Generator):
        classes = rng.choice(self.num_classes, size=n_way, replace=False)
        support_x, support_y, query_x, query_y = [], [], [], []
        for i, c in enumerate(classes):
            idxs = self.by_class[c]
            chosen = rng.choice(len(idxs), size=k_shot + q_query, replace=False)
            sup_idx = [idxs[j] for j in chosen[:k_shot]]
            qry_idx = [idxs[j] for j in chosen[k_shot:]]
            for si in sup_idx:
                x, _ = self.base[si]
                support_x.append(x)
                support_y.append(i)
            for qi in qry_idx:
                x, _ = self.base[qi]
                query_x.append(x)
                query_y.append(i)
        sx = torch.stack(support_x, dim=0)
        sy = torch.tensor(support_y, dtype=torch.long)
        qx = torch.stack(query_x, dim=0)
        qy = torch.tensor(query_y, dtype=torch.long)
        return sx, sy, qx, qy


@dataclass
class MAMLConfig:
    dataset: str = "mnist"
    n_way: int = 5
    k_shot: int = 1
    q_query: int = 15
    inner_steps: int = 1
    inner_lr: float = 0.4
    second_order: bool = False
    meta_lr: float = 1e-3
    episodes_per_epoch: int = 200
    epochs: int = 20
    batch_size_episodes: int = 4
    base_filters: int = 32
    depth: int = 4
    kernel_size: int = 3
    dropout: float = 0.0
    use_bn: bool = True
    seed: int = 42
    data_dir: str = "./data"
    out_dir: str = "./results_maml"
    num_workers: int = 2

def clone_params(named_params: Dict[str, torch.Tensor]):
    return {k: v.clone() for k, v in named_params.items()}

def functional_forward(model: nn.Module, params: Dict[str, torch.Tensor], x: torch.Tensor):
    from torch.func import functional_call
    return functional_call(model, params, (x,))

def inner_adapt(model, support_x, support_y, fast_params, inner_lr, second_order):
    for _ in range(args.inner_steps):
        logits = functional_forward(model, fast_params, support_x)
        loss = F.cross_entropy(logits, support_y)
        grads = torch.autograd.grad(
            loss, list(fast_params.values()),
            create_graph=second_order, retain_graph=second_order
        )
        for (name, w), g in zip(fast_params.items(), grads):
            fast_params[name] = w - inner_lr * g
    return fast_params


def main():
    parser = argparse.ArgumentParser("MAML on CNN (few-shot)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_bn", action="store_true")
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--inner_lr", type=float, default=0.4)
    parser.add_argument("--second_order", action="store_true", help="Enable full MAML (else FOMAML)")
    parser.add_argument("--meta_lr", type=float, default=1e-3)
    parser.add_argument("--episodes_per_epoch", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size_episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./results_maml")
    parser.add_argument("--num_workers", type=int, default=2)

    global args
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
        mean, std, in_ch = (0.1307,), (0.3081,), 1
        n_classes = 10
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        base_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=t)
        base_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=t)
    else:
        mean, std, in_ch = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 3
        n_classes = 10
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        t_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        base_train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=t_train)
        base_test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=t_test)

    train_ci = ClassIndexedDataset(base_train, n_classes)
    test_ci = ClassIndexedDataset(base_test, n_classes)

    model = FewShotCNN(
        in_channels=in_ch,
        base_filters=args.base_filters,
        depth=args.depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_bn=args.use_bn,
        n_way=args.n_way
    ).to(device)
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.dataset}_{args.n_way}way_{args.k_shot}shot_q{args.q_query}_inner{args.inner_steps}-lr{args.inner_lr}_" \
               f"{'2nd' if args.second_order else '1st'}_mf{args.base_filters}_d{args.depth}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "dataset": args.dataset,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "q_query": args.q_query,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "second_order": args.second_order,
        "meta_lr": args.meta_lr,
        "episodes_per_epoch": args.episodes_per_epoch,
        "epochs": args.epochs,
        "batch_size_episodes": args.batch_size_episodes,
        "base_filters": args.base_filters,
        "depth": args.depth,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "use_bn": args.use_bn,
        "seed": args.seed,
        "device": str(device),
        "run_name": run_name
    }
    save_json(cfg, run_dir / "config.json")

    rng = np.random.default_rng(args.seed)
    metrics = []
    best_val = 0.0
    start = time.time()

    def meta_batch(phase: str):
        episodes = []
        ci = train_ci if phase == "train" else test_ci
        for _ in range(args.batch_size_episodes):
            sx, sy, qx, qy = ci.sample_episode(args.n_way, args.k_shot, args.q_query, rng)
            episodes.append((
                sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            ))
        return episodes

    for epoch in range(1, args.epochs + 1):
        model.train()
        epi_losses, epi_accs = [], []
        for _ in range(args.episodes_per_epoch):
            episodes = meta_batch("train")
            meta_opt.zero_grad()
            outer_loss = 0.0
            outer_acc = 0.0

            for support_x, support_y, query_x, query_y in episodes:
                named_params = dict(model.named_parameters())
                fast_params = clone_params(named_params)
                fast_params = inner_adapt(model, support_x, support_y, fast_params, args.inner_lr, args.second_order)
                q_logits = functional_forward(model, fast_params, query_x)
                q_loss = F.cross_entropy(q_logits, query_y)
                q_acc = accuracy_from_logits(q_logits.detach(), query_y)
                q_loss.backward()
                outer_loss += q_loss.item()
                outer_acc += q_acc

            meta_opt.step()

            epi_losses.append(outer_loss / args.batch_size_episodes)
            epi_accs.append(outer_acc / args.batch_size_episodes)

        model.eval()
        val_accs = []
        for sx, sy, qx, qy in [test_ci.sample_episode(args.n_way, args.k_shot, args.q_query, rng)
                               for _ in range(max(50, args.batch_size_episodes))]:
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            with torch.enable_grad():
                named_params = dict(model.named_parameters())
                fast_params = clone_params(named_params)
                fast_params = inner_adapt(model, sx, sy, fast_params, args.inner_lr, False)
            with torch.no_grad():
                q_logits = functional_forward(model, fast_params, qx)
                val_accs.append(accuracy_from_logits(q_logits, qy))
        val_acc = float(np.mean(val_accs))

        log = {
            "epoch": epoch,
            "train_outer_loss": float(np.mean(epi_losses)),
            "train_outer_acc": float(np.mean(epi_accs)),
            "val_acc": val_acc
        }
        metrics.append(log)
        save_json(metrics, run_dir / "metrics.json")
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_outer_loss={log['train_outer_loss']:.4f} "
              f"train_outer_acc={log['train_outer_acc']:.4f} "
              f"| val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), run_dir / "best.pt")

    elapsed = time.time() - start
    summary = {
        "best_val_acc": best_val,
        "train_time_sec": elapsed,
        "params": sum(p.numel() for p in model.parameters()),
        "episodes_seen": args.episodes_per_epoch * args.epochs
    }
    print("\n== Summary ==")
    for k, v in summary.items():
        print(f"{k}: {v}")
    save_json(summary, run_dir / "summary.json")

if __name__ == "__main__":
    main()
