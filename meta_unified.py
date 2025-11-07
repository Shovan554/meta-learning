import argparse, json, csv, random, time, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


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


def clone_params(named_params: Dict[str, torch.Tensor]):
    return {k: v.clone() for k, v in named_params.items()}

def functional_forward(model: nn.Module, params: Dict[str, torch.Tensor], x: torch.Tensor):
    from torch.func import functional_call
    return functional_call(model, params, (x,))


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
    def __init__(self, hidden=32, use_lstm=False):
        super().__init__()
        self.use_lstm = use_lstm
        in_dim = 3
        if use_lstm:
            self.cell = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, 2)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 2)
            )
        self.softplus = nn.Softplus()

    def forward(self, feat, hx=None):
        if self.use_lstm:
            out, hx = self.cell(feat, hx)
            param = self.head(out)
        else:
            B, T, D = feat.shape
            flat = feat.view(B*T, D)
            param = self.mlp(flat).view(B, T, 2)
            hx = None
        step_raw, mom_raw = param[..., 0], param[..., 1]
        step = self.softplus(step_raw)
        momentum = torch.sigmoid(mom_raw)
        return step, momentum, hx

def grad_features(t: torch.Tensor, eps=1e-8):
    g = t
    mean_abs = g.abs().mean()
    std = g.std(unbiased=False)
    norm = torch.linalg.vector_norm(g) / (g.numel() ** 0.5 + eps)
    return torch.stack([mean_abs, std, norm])

def inner_adapt_lopt(model, support_x, support_y, fast_params, opt_net: OptimizerNet,
                     inner_steps, state_dict: Dict[str, torch.Tensor]):
    for _ in range(inner_steps):
        logits = functional_forward(model, fast_params, support_x)
        loss = F.cross_entropy(logits, support_y)
        grads = torch.autograd.grad(loss, list(fast_params.values()), create_graph=True)

        feats = torch.stack([grad_features(g.detach()) for g in grads], dim=0)[None, ...]
        step_scales, momenta, _ = opt_net(feats)
        step_scales, momenta = step_scales.squeeze(0), momenta.squeeze(0)

        for (name, w), g, step, mom in zip(list(fast_params.items()), grads, step_scales, momenta):
            mkey = name + "_m"
            if mkey not in state_dict:
                state_dict[mkey] = torch.zeros_like(w)
            m = state_dict[mkey]
            m = mom * m + (1 - mom) * g
            state_dict[mkey] = m
            fast_params[name] = w - step * m
    return fast_params, state_dict


def run_meta(
    approach="maml",
    dataset="mnist", n_way=5, k_shot=1, q_query=15,
    inner_steps=1, inner_lr=0.4, second_order=False,
    meta_lr=1e-3, episodes_per_epoch=200, epochs=10, batch_size_episodes=4,
    base_filters=32, depth=4, kernel_size=3, dropout=0.0, use_bn=True,
    seed=42, data_dir="./data", device=None, rng=None,
    lopt_hidden=32, lopt_lstm=False
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = rng or np.random.default_rng(seed)

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
        meta_opt = torch.optim.Adam(model.parameters(), lr=meta_lr)
        opt_net = None
        lopt_state = None
    else:
        opt_net = OptimizerNet(hidden=lopt_hidden, use_lstm=lopt_lstm).to(device)
        meta_opt = torch.optim.Adam(list(model.parameters()) + list(opt_net.parameters()), lr=meta_lr)
        lopt_state = {}

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
        val_accs = []
        for _ in range(max(50, batch_size_episodes)):
            sx, sy, qx, qy = test_ci.sample_episode(n_way, k_shot, q_query, rng)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            named = dict(model.named_parameters()); fast = clone_params(named)
            if approach == "maml":
                with torch.enable_grad():
                    fast = inner_adapt_maml(model, sx, sy, fast, inner_lr, inner_steps, False)
            else:
                with torch.enable_grad():
                    fast, _ = inner_adapt_lopt(model, sx, sy, fast, opt_net, inner_steps, {})
            with torch.no_grad():
                q_logits = functional_forward(model, fast, qx)
                val_accs.append(acc_from_logits(q_logits, qy))
        val_acc = float(np.mean(val_accs))

        log = {
            "epoch": epoch,
            "train_outer_loss": float(np.mean(train_losses)),
            "train_outer_acc":  float(np.mean(train_accs)),
            "val_acc": val_acc
        }
        metrics.append(log)
        print(f"[{approach}][{epoch:03d}/{epochs}] "
              f"train_outer_loss={log['train_outer_loss']:.4f} "
              f"train_outer_acc={log['train_outer_acc']:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val: best_val = val_acc

    elapsed = time.time() - start
    summary = {
        "approach": approach,
        "best_val_acc": best_val,
        "train_time_sec": elapsed,
        "params_backbone": sum(p.numel() for p in model.parameters()),
        "params_total": sum(p.numel() for p in model.parameters()) + (sum(p.numel() for p in opt_net.parameters()) if approach=="lopt" else 0),
        "episodes_seen": episodes_per_epoch * epochs,
        "inner_steps": inner_steps,
        "inner_lr": inner_lr if approach=="maml" else None,
    }
    last = metrics[-1]
    gap = last["train_outer_acc"] - last["val_acc"]
    summary["overfit_gap"] = gap
    summary["overfit_flag"] = gap >= 0.10
    return metrics, summary

def ablation_sweep(args, device):
    set_seed(args.seed); rng = np.random.default_rng(args.seed)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.n_way}way"
    run_dir = out_root / f"{tag}_ABLATE_{args.approach}"
    run_dir.mkdir(parents=True, exist_ok=True)

    inner_steps_list = [int(x) for x in args.inner_steps_sweep.split(",")]
    support_list = [int(x) for x in args.support_sweep.split(",")]
    query_list = [int(x) for x in args.query_sweep.split(",")]

    rows = []
    for s in inner_steps_list:
        for k in support_list:
            for q in query_list:
                print(f"\n=== {args.approach} | inner_steps={s} | k_shot={k} | q_query={q} ===")
                metrics, summary = run_meta(
                    approach=args.approach,
                    dataset=args.dataset, n_way=args.n_way, k_shot=k, q_query=q,
                    inner_steps=s, inner_lr=args.inner_lr, second_order=args.second_order,
                    meta_lr=args.meta_lr, episodes_per_epoch=args.episodes_per_epoch,
                    epochs=args.epochs, batch_size_episodes=args.batch_size_episodes,
                    base_filters=args.base_filters, depth=args.depth, kernel_size=args.kernel_size,
                    dropout=args.dropout, use_bn=args.use_bn, seed=args.seed,
                    data_dir=args.data_dir, device=device,
                    lopt_hidden=args.lopt_hidden, lopt_lstm=args.lopt_lstm
                )
                cell = run_dir / f"steps{s}_k{k}_q{q}"
                cell.mkdir(parents=True, exist_ok=True)
                save_json(vars(args) | {"inner_steps": s, "k_shot": k, "q_query": q}, cell / "config.json")
                save_json(metrics, cell / "metrics.json")
                save_json(summary, cell / "summary.json")

                row = {
                    "approach": args.approach,
                    "inner_steps": s, "k_shot": k, "q_query": q,
                    "best_val_acc": summary["best_val_acc"],
                    "overfit_gap": summary["overfit_gap"],
                    "overfit_flag": summary["overfit_flag"],
                    "episodes_seen": summary["episodes_seen"],
                    "train_time_sec": summary["train_time_sec"],
                    "params_total": summary["params_total"]
                }
                rows.append(row)
                print(f"-> best_val_acc={row['best_val_acc']:.4f} | overfit_gap={row['overfit_gap']:.3f}")

    write_csv(rows, run_dir / "ablation_results.csv")
    save_json(rows, run_dir / "ablation_results.json")
    print("\n== Ablation complete ==")
    for r in rows:
        print(f"{r['approach']} steps={r['inner_steps']} k={r['k_shot']} q={r['q_query']} "
              f"-> val={r['best_val_acc']:.4f} gap={r['overfit_gap']:.3f}")

def main():
    p = argparse.ArgumentParser("Unified Meta-Learning (MAML + Learned Optimizer) with Ablations")
    p.add_argument("--approach", choices=["maml","lopt"], default="lopt")
    p.add_argument("--dataset", choices=["mnist","cifar10"], default="mnist")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--base_filters", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_bn", action="store_true")
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument("--second_order", action="store_true", help="Full MAML grads (MAML only)")
    p.add_argument("--inner_lr", type=float, default=0.4, help="MAML only")
    p.add_argument("--meta_lr", type=float, default=1e-3)
    p.add_argument("--episodes_per_epoch", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size_episodes", type=int, default=4)
    p.add_argument("--lopt_hidden", type=int, default=32)
    p.add_argument("--lopt_lstm", action="store_true")
    p.add_argument("--do_ablate", action="store_true")
    p.add_argument("--inner_steps_sweep", type=str, default="1,2")
    p.add_argument("--support_sweep", type=str, default="1,5")
    p.add_argument("--query_sweep", type=str, default="5,15")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./results_unified")

    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.do_ablate:
        ablation_sweep(args, device)
    else:
        metrics, summary = run_meta(
            approach=args.approach,
            dataset=args.dataset, n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
            inner_steps=args.inner_steps, inner_lr=args.inner_lr, second_order=args.second_order,
            meta_lr=args.meta_lr, episodes_per_epoch=args.episodes_per_epoch,
            epochs=args.epochs, batch_size_episodes=args.batch_size_episodes,
            base_filters=args.base_filters, depth=args.depth, kernel_size=args.kernel_size,
            dropout=args.drop_out if hasattr(args,'drop_out') else args.dropout, use_bn=args.use_bn,
            seed=args.seed, data_dir=args.data_dir, device=device,
            lopt_hidden=args.lopt_hidden, lopt_lstm=args.lopt_lstm
        )
        out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
        run_tag = f"{args.approach}_{args.dataset}_{args.n_way}way_{args.k_shot}shot_q{args.q_query}_steps{args.inner_steps}"
        run_dir = out_root / run_tag; run_dir.mkdir(parents=True, exist_ok=True)
        save_json(vars(args) | {"device": str(device)}, run_dir / "config.json")
        save_json(metrics, run_dir / "metrics.json")
        save_json(summary, run_dir / "summary.json")
        print("\n== Summary =="); [print(f"{k}: {v}") for k,v in summary.items()]

if __name__ == "__main__":
    main()
