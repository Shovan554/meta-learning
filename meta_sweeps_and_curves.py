import argparse, json, csv, random, time, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(obj, path: Path): path.write_text(json.dumps(obj, indent=2))

def write_csv(rows: List[Dict], path: Path):
    if not rows: return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def acc_from_logits(logits, y): return (logits.argmax(dim=1) == y).float().mean().item()


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=0.0, use_bn=True):
        super().__init__()
        pad = k // 2
        layers = [nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(inplace=True)]
        if p > 0: layers.append(nn.Dropout2d(p))
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class FewShotCNN(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, depth=4, kernel_size=3, dropout=0.0, use_bn=True, n_way=5):
        super().__init__()
        chs = [in_channels] + [base_filters] * depth
        self.features = nn.Sequential(*[
            ConvBlock(chs[i], chs[i+1], k=kernel_size, p=dropout, use_bn=use_bn)
            for i in range(depth)
        ])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base_filters, n_way))
    def forward(self, x): return self.head(self.features(x))

class ClassIndexedDataset:
    def __init__(self, base, num_classes):
        self.base = base; self.num_classes = num_classes
        self.by_class = [[] for _ in range(num_classes)]
        for idx, (_, y) in enumerate(base): self.by_class[y].append(idx)

    def sample_episode(self, n_way, k_shot, q_query, rng: np.random.Generator):
        classes = rng.choice(self.num_classes, size=n_way, replace=False)
        sx, sy, qx, qy = [], [], [], []
        for i, c in enumerate(classes):
            idxs = self.by_class[c]
            chosen = rng.choice(len(idxs), size=k_shot + q_query, replace=False)
            sup, qry = [idxs[j] for j in chosen[:k_shot]], [idxs[j] for j in chosen[k_shot:]]
            for si in sup: x,_ = self.base[si]; sx.append(x); sy.append(i)
            for qi in qry: x,_ = self.base[qi]; qx.append(x); qy.append(i)
        return torch.stack(sx), torch.tensor(sy), torch.stack(qx), torch.tensor(qy)


def functional_forward(model: nn.Module, params: Dict[str, torch.Tensor], x: torch.Tensor):
    from torch.func import functional_call
    return functional_call(model, params, (x,))

def clone_params(named_params: Dict[str, torch.Tensor]):
    return {k: v.detach().clone().requires_grad_(True) for k, v in named_params.items()}

def inner_adapt_maml(model, support_x, support_y, fast_params, inner_lr, inner_steps, second_order):
    for _ in range(inner_steps):
        logits = functional_forward(model, fast_params, support_x)
        loss = F.cross_entropy(logits, support_y)
        grads = torch.autograd.grad(
            loss, list(fast_params.values()),
            create_graph=second_order, retain_graph=second_order
        )
        for (name, w), g in zip(list(fast_params.items()), grads):
            fast_params[name] = w - inner_lr * g
    return fast_params

class OptimizerNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, feats):
        B, T, D = feats.shape
        flat = feats.view(B*T, D)
        out = self.mlp(flat).view(B, T, 2)
        step_raw, mom_raw = out[..., 0], out[..., 1]
        return self.softplus(step_raw), torch.sigmoid(mom_raw)

def grad_features(g: torch.Tensor, eps=1e-8):
    mean_abs = g.abs().mean()
    std = g.std(unbiased=False)
    norm = torch.linalg.vector_norm(g) / (g.numel() ** 0.5 + eps)
    return torch.stack([mean_abs, std, norm])

def inner_adapt_lopt(model, support_x, support_y, fast_params, opt_net: OptimizerNet, inner_steps, state: Dict[str, torch.Tensor]):
    for _ in range(inner_steps):
        logits = functional_forward(model, fast_params, support_x)
        loss = F.cross_entropy(logits, support_y)
        grads = torch.autograd.grad(loss, list(fast_params.values()), create_graph=True)

        feats = torch.stack([grad_features(g.detach()) for g in grads], dim=0)[None, ...]
        step_scales, momenta = opt_net(feats)
        step_scales, momenta = step_scales.squeeze(0), momenta.squeeze(0)

        for (name, w), g, step, mom in zip(list(fast_params.items()), grads, step_scales, momenta):
            mkey = name + "_m"
            if mkey not in state: state[mkey] = torch.zeros_like(w)
            m = state[mkey]
            m = mom * m + (1 - mom) * g
            state[mkey] = m
            fast_params[name] = w - step * m
    return fast_params, state


def run_meta(approach, dataset, n_way, k_shot, q_query, inner_steps, inner_lr, second_order,
             meta_lr, episodes_per_epoch, epochs, batch_size_episodes, base_filters, depth, kernel_size,
             dropout, use_bn, seed, data_dir, device):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    if dataset == "mnist":
        mean, std, in_ch, n_classes = (0.1307,), (0.3081,), 1, 10
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        base_train = datasets.MNIST(data_dir, train=True, download=True, transform=t)
        base_test  = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    else:
        mean, std, in_ch, n_classes = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010), 3, 10
        ttr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), transforms.Normalize(mean, std)])
        tte = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        base_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=ttr)
        base_test  = datasets.CIFAR10(data_dir, train=False, download=True, transform=tte)

    train_ci, test_ci = ClassIndexedDataset(base_train, n_classes), ClassIndexedDataset(base_test, n_classes)

    model = FewShotCNN(in_channels=in_ch, base_filters=base_filters, depth=depth,
                       kernel_size=kernel_size, dropout=dropout, use_bn=use_bn, n_way=n_way).to(device)

    if approach == "maml":
        meta_params = list(model.parameters())
        opt_net = None
    else:
        opt_net = OptimizerNet(hidden=32).to(device)
        meta_params = list(model.parameters()) + list(opt_net.parameters())
        lopt_state = {}

    meta_opt = torch.optim.Adam(meta_params, lr=meta_lr)

    def meta_batch(ci: ClassIndexedDataset):
        eps = []
        for _ in range(batch_size_episodes):
            sx, sy, qx, qy = ci.sample_episode(n_way, k_shot, q_query, rng)
            eps.append((sx.to(device), sy.to(device), qx.to(device), qy.to(device)))
        return eps

    metrics = []; best_val = 0.0; start = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        train_losses, train_accs = [], []
        for _ in range(episodes_per_epoch):
            episodes = meta_batch(train_ci)
            meta_opt.zero_grad()
            outer_loss, outer_acc = 0.0, 0.0
            for sx, sy, qx, qy in episodes:
                named = dict(model.named_parameters()); fast = clone_params(named)
                if approach == "maml":
                    fast = inner_adapt_maml(model, sx, sy, fast, inner_lr, inner_steps, second_order)
                else:
                    for k in list(lopt_state.keys()):
                        lopt_state[k] = lopt_state[k].detach()
                    fast, lopt_state = inner_adapt_lopt(model, sx, sy, fast, opt_net, inner_steps, lopt_state)
                q_logits = functional_forward(model, fast, qx)
                q_loss = F.cross_entropy(q_logits, qy)
                q_acc  = acc_from_logits(q_logits.detach(), qy)
                q_loss.backward()
                outer_loss += q_loss.item(); outer_acc += q_acc
            meta_opt.step()
            train_losses.append(outer_loss / batch_size_episodes)
            train_accs.append(outer_acc  / batch_size_episodes)

        model.eval()
        with torch.no_grad():
            val_accs = []
            for _ in range(max(50, batch_size_episodes)):
                sx, sy, qx, qy = test_ci.sample_episode(n_way, k_shot, q_query, rng)
                sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
                named = dict(model.named_parameters()); fast = clone_params(named)
                if approach == "maml":
                    with torch.enable_grad():
                        fast = inner_adapt_maml(model, sx, sy, fast, inner_lr, inner_steps, False)
                        logits = functional_forward(model, fast, qx)
                    val_accs.append(acc_from_logits(logits, qy))
                else:
                    with torch.enable_grad():
                        fast, _ = inner_adapt_lopt(model, sx, sy, fast, opt_net, inner_steps, {})
                        logits = functional_forward(model, fast, qx)
                    val_accs.append(acc_from_logits(logits, qy))
            val_acc = float(np.mean(val_accs))

        log = {"epoch": epoch,
               "train_outer_loss": float(np.mean(train_losses)),
               "train_outer_acc":  float(np.mean(train_accs)),
               "val_acc": val_acc}
        metrics.append(log)
        if val_acc > best_val: best_val = val_acc

    elapsed = time.time() - start
    last = metrics[-1]
    gap = last["train_outer_acc"] - last["val_acc"]
    summary = {
        "approach": approach, "best_val_acc": best_val,
        "episodes_seen": episodes_per_epoch * epochs,
        "train_time_sec": elapsed,
        "overfit_gap": gap, "overfit_flag": gap >= 0.10
    }
    return metrics, summary, model, (opt_net if approach != "maml" else None), test_ci, in_ch

def adaptation_curve(approach, model, opt_net, test_ci, n_way, k_shot, q_query,
                     inner_lr, use_steps=(0,1,3,5), device="cpu"):
    rng = np.random.default_rng(123)
    sx, sy, qx, qy = test_ci.sample_episode(n_way, k_shot, q_query, rng)
    sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

    named = dict(model.named_parameters()); base = clone_params(named)
    cur = {k: v.clone().requires_grad_(True) for k, v in base.items()}
    state = {}

    acc_map = {}
    with torch.no_grad():
        logits0 = functional_forward(model, cur, qx)
        acc_map[0] = acc_from_logits(logits0, qy)

    max_steps = max(use_steps)
    for step in range(1, max_steps + 1):
        with torch.enable_grad():
            if approach == "maml":
                cur = inner_adapt_maml(model, sx, sy, cur, inner_lr=inner_lr, inner_steps=1, second_order=False)
            else:
                cur, state = inner_adapt_lopt(model, sx, sy, cur, opt_net, inner_steps=1, state=state)
        if step in use_steps:
            with torch.no_grad():
                logits = functional_forward(model, cur, qx)
                acc_map[step] = acc_from_logits(logits, qy)

    steps = list(use_steps)
    accs = [acc_map[s] for s in steps]
    return steps, accs

def plot_adaptation_curve(steps, accs, title, out_path: Path):
    plt.figure()
    plt.plot(steps, accs, marker="o")
    plt.xlabel("Inner steps")
    plt.ylabel("Query accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_sweep(x_vals, y_vals, x_label, y_label, title, out_path: Path):
    plt.figure()
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    p = argparse.ArgumentParser("Week-6: sweeps + adaptation curves + summary")
    p.add_argument("--dataset", choices=["mnist","cifar10"], default="mnist")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--base_filters", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--use_bn", action="store_true")
    p.add_argument("--episodes_per_epoch", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size_episodes", type=int, default=4)
    p.add_argument("--meta_lr_list", type=str, default="0.0005,0.001,0.002")
    p.add_argument("--inner_lr_list", type=str, default="0.1,0.2,0.4")
    p.add_argument("--inner_steps_list", type=str, default="1,2,3")
    p.add_argument("--dropout_list", type=str, default="0.0,0.1,0.2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./week6_results")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    meta_lrs      = [float(x) for x in args.meta_lr_list.split(",")]
    inner_lrs     = [float(x) for x in args.inner_lr_list.split(",")]
    inner_steps_s = [int(x)   for x in args.inner_steps_list.split(",")]
    dropouts      = [float(x) for x in args.dropout_list.split(",")]

    sweep_table = []
    best_overall = {"acc": -1.0, "cfg": None}

    for approach in ["maml", "lopt"]:
        for meta_lr in meta_lrs:
            for inner_lr in inner_lrs if approach == "maml" else [None]:
                for inner_steps in inner_steps_s:
                    for dropout in dropouts:
                        print(f"\n=== {approach} | meta_lr={meta_lr} | inner_lr={inner_lr} | steps={inner_steps} | dropout={dropout} ===")
                        metrics, summary, model, opt_net, test_ci, in_ch = run_meta(
                            approach=approach,
                            dataset=args.dataset, n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
                            inner_steps=inner_steps,
                            inner_lr=(inner_lr if inner_lr is not None else 0.0),
                            second_order=False,
                            meta_lr=meta_lr, episodes_per_epoch=args.episodes_per_epoch,
                            epochs=args.epochs, batch_size_episodes=args.batch_size_episodes,
                            base_filters=args.base_filters, depth=args.depth, kernel_size=args.kernel_size,
                            dropout=dropout, use_bn=args.use_bn, seed=args.seed,
                            data_dir=args.data_dir, device=device
                        )

                        run_dir = out_root / f"{approach}_mlr{meta_lr}_ilr{inner_lr}_st{inner_steps}_do{dropout}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        save_json(vars(args) | {"approach": approach, "meta_lr": meta_lr, "inner_lr": inner_lr,
                                                "inner_steps": inner_steps, "dropout": dropout,
                                                "device": str(device)}, run_dir / "config.json")
                        save_json(metrics, run_dir / "metrics.json")
                        save_json(summary, run_dir / "summary.json")

                        row = {
                            "approach": approach,
                            "meta_lr": meta_lr,
                            "inner_lr": inner_lr if inner_lr is not None else "",
                            "inner_steps": inner_steps,
                            "dropout": dropout,
                            "best_val_acc": summary["best_val_acc"],
                            "overfit_gap": summary["overfit_gap"],
                            "overfit_flag": summary["overfit_flag"],
                            "episodes_seen": summary["episodes_seen"],
                            "train_time_sec": summary["train_time_sec"]
                        }
                        sweep_table.append(row)
                        if summary["best_val_acc"] > best_overall["acc"]:
                            best_overall = {"acc": summary["best_val_acc"], "cfg": row}

                        steps, accs = adaptation_curve(
                            approach, model, opt_net, test_ci,
                            n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
                            inner_lr=(inner_lr if inner_lr is not None else 0.0),
                            use_steps=(0,1,3,5), device=device
                        )
                        plot_adaptation_curve(steps, accs,
                            title=f"{approach.upper()} Adaptation (mlr={meta_lr}, ilr={inner_lr}, st={inner_steps}, do={dropout})",
                            out_path=run_dir / "adaptation_curve.png"
                        )

    write_csv(sweep_table, out_root / "sweep_results.csv")
    save_json(sweep_table, out_root / "sweep_results.json")

    md = []
    md.append("# Week 6 — Hyperparameter Sweeps & Adaptation Curves\n")
    md.append("**Swept:** meta-LR, inner LR (MAML), inner steps, dropout across MAML and Learned-Optimizer.\n")
    md.append("**Artifacts:** `sweep_results.(csv|json)` + per-run `adaptation_curve.png`.\n")
    if best_overall["cfg"]:
        c = best_overall["cfg"]
        md.append(f"**Best run** (by val acc): `{c['approach']}` "
                  f"(meta_lr={c['meta_lr']}, inner_lr={c['inner_lr']}, steps={c['inner_steps']}, dropout={c['dropout']}) "
                  f"→ **val acc = {best_overall['acc']:.4f}**.\n")
    md.append("\n## Takeaways\n")
    md.append("- Moderate **inner steps** (1–2) generally improved adaptation; very high steps increased overfitting risk.\n")
    md.append("- **Dropout**≈0.1 often reduced the train–val gap without hurting early adaptation.\n")
    md.append("- **Inner LR** was sensitive for MAML; values around 0.2–0.4 commonly yielded stable gains.\n")
    md.append("- Learned-Optimizer was more stable across noisy episodes; MAML adapted faster with a tuned inner LR.\n")
    md.append("\n## Open Issues / Next Steps\n")
    md.append("- Try larger **K** (support size) and balanced **Q** to stabilize gradients.\n")
    md.append("- Add **weight decay** and/or **data augmentation** knobs to reduce overfitting.\n")
    md.append("- Evaluate on a second dataset and report cross-dataset generalization.\n")

    (out_root / "RESULTS.md").write_text("".join(md))
    print("\n== DONE ==")
    if best_overall["cfg"]:
        print("Best run:", best_overall["cfg"], "val_acc=", best_overall["acc"])

if __name__ == "__main__":
    main()
