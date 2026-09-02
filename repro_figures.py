"""Canonical Net2Net proxy figures: function preservation and grown-vs-scratch."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from device_utils import seed_everything, select_device
from net2net import deeper, wider

try:
    import wandb as wandb_lib
except ImportError:
    wandb_lib = None

REPO_ROOT = Path(__file__).resolve().parent


class MnistNet(nn.Module):
    """LeNet-style MNIST net used by ``examples/train_mnist.py``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CifarNet(nn.Module):
    """Compact CIFAR convnet used by ``examples/train_cifar10.py``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(5, 1)
        self.fc1 = nn.Linear(32 * 3 * 3, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, np.sqrt(2.0 / n))
                module.bias.data.fill_(0.0)
            if isinstance(module, nn.Linear):
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        return F.log_softmax(self.fc1(x), dim=1)


def function_delta(before: torch.Tensor, after: torch.Tensor) -> dict[str, float]:
    """Return max/mean absolute output change after a growth operator."""
    diff = (after - before).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "l2": float(torch.linalg.vector_norm(after - before).item()),
    }


def apply_mnist_wider(net: MnistNet, *, noise: bool) -> MnistNet:
    """Widen conv1→15 and conv2→30, matching the example script."""
    net.conv1, net.conv2, _ = wider(
        net.conv1,
        net.conv2,
        15,
        noise=noise,
        random_init=False,
        weight_norm=False,
    )
    net.conv2, net.fc1, _ = wider(
        net.conv2,
        net.fc1,
        30,
        noise=noise,
        random_init=False,
        weight_norm=False,
    )
    return net


def apply_mnist_deeper(net: MnistNet, *, noise: bool) -> MnistNet:
    """Insert identity conv layers after conv1 and conv2."""
    net.conv1 = deeper(
        net.conv1, nn.ReLU, bnorm_flag=False, weight_norm=False, noise=noise
    )
    net.conv2 = deeper(
        net.conv2, nn.ReLU, bnorm_flag=False, weight_norm=False, noise=noise
    )
    return net


def apply_cifar_wider(net: CifarNet, *, noise: bool) -> CifarNet:
    """Widen 8→12, 16→24, 32→48 with matching BN, as in the CIFAR example."""
    net.conv1, net.conv2, net.bn1 = wider(
        net.conv1,
        net.conv2,
        12,
        net.bn1,
        noise=noise,
        random_init=False,
        weight_norm=False,
    )
    net.conv2, net.conv3, net.bn2 = wider(
        net.conv2,
        net.conv3,
        24,
        net.bn2,
        noise=noise,
        random_init=False,
        weight_norm=False,
    )
    net.conv3, net.fc1, net.bn3 = wider(
        net.conv3,
        net.fc1,
        48,
        net.bn3,
        noise=noise,
        random_init=False,
        weight_norm=False,
    )
    return net


def apply_cifar_deeper(net: CifarNet, *, noise: bool) -> CifarNet:
    """Insert identity convs after each CIFAR conv (BN identity)."""
    net.conv1 = deeper(
        net.conv1, nn.ReLU, bnorm_flag=True, weight_norm=False, noise=noise
    )
    net.conv2 = deeper(
        net.conv2, nn.ReLU, bnorm_flag=True, weight_norm=False, noise=noise
    )
    net.conv3 = deeper(
        net.conv3, nn.ReLU, bnorm_flag=True, weight_norm=False, noise=noise
    )
    return net


def make_scratch_mnist_wider() -> MnistNet:
    """Target-width MNIST net trained from scratch."""
    net = MnistNet()
    net.conv1 = nn.Conv2d(1, 15, kernel_size=5)
    net.conv2 = nn.Conv2d(15, 30, kernel_size=5)
    net.fc1 = nn.Linear(480, 50)
    return net


def make_scratch_cifar_wider() -> CifarNet:
    """Target-width CIFAR net trained from scratch."""
    net = CifarNet()
    net.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
    net.bn1 = nn.BatchNorm2d(12)
    net.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
    net.bn2 = nn.BatchNorm2d(24)
    net.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
    net.bn3 = nn.BatchNorm2d(48)
    net.fc1 = nn.Linear(48 * 3 * 3, 10)
    return net


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _mnist_loaders(data_dir: Path, batch_size: int, device: torch.device) -> tuple[DataLoader, DataLoader]:
    kwargs: dict[str, Any] = (
        {"num_workers": 0, "pin_memory": True} if device.type == "cuda" else {}
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = DataLoader(
        datasets.MNIST(str(data_dir), train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = DataLoader(
        datasets.MNIST(str(data_dir), train=False, transform=transform),
        batch_size=1000,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader


def _cifar_loaders(data_dir: Path, batch_size: int, device: torch.device) -> tuple[DataLoader, DataLoader]:
    kwargs: dict[str, Any] = (
        {"num_workers": 0, "pin_memory": True} if device.type == "cuda" else {}
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_loader = DataLoader(
        datasets.CIFAR10(str(data_dir), train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = DataLoader(
        datasets.CIFAR10(str(data_dir), train=False, transform=transform),
        batch_size=1000,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Return (accuracy, average nll) on ``loader``."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += int(pred.eq(target).sum().item())
            n += target.size(0)
    return correct / n, loss_sum / n


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
) -> dict[str, list[float]]:
    """SGD train; record train/test accuracy each epoch."""
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    history: dict[str, list[float]] = {
        "train_acc": [],
        "test_acc": [],
        "train_loss": [],
        "test_loss": [],
    }
    for _epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        n = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * target.size(0)
            pred = output.argmax(dim=1)
            correct += int(pred.eq(target).sum().item())
            n += target.size(0)
        test_acc, test_loss = evaluate(model, test_loader, device)
        history["train_acc"].append(correct / n)
        history["train_loss"].append(loss_sum / n)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        print(
            f"epoch {_epoch + 1}/{epochs} train_acc={correct / n:.4f} "
            f"test_acc={test_acc:.4f}",
            flush=True,
        )
    return history


def measure_growth_deltas(
    make_net: Any,
    apply_wider: Any,
    apply_deeper: Any,
    sample: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Function-preservation error at growth (noise off)."""
    results: dict[str, dict[str, float]] = {}
    for name, apply_op in (("wider", apply_wider), ("deeper", apply_deeper)):
        net = make_net().to(device).eval()
        with torch.no_grad():
            before = net(sample)
        apply_op(net, noise=False)
        net.eval()
        with torch.no_grad():
            after = net(sample)
        results[name] = function_delta(before, after)
    return results


def plot_function_preservation(
    rows: dict[str, dict[str, dict[str, float]]], out_path: Path
) -> None:
    """Bar chart of max-abs output error at growth, per dataset and operator."""
    labels = []
    values = []
    for dataset, ops in rows.items():
        for op, delta in ops.items():
            labels.append(f"{dataset}\n{op}")
            values.append(delta["max_abs"])
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(range(len(values)), values, color="#4c78a8")
    ax.axhline(1e-4, color="crimson", linestyle="--", label="1e-4 gate")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("max |Δoutput| at growth (noise=0)")
    ax.set_title("Function preservation at Net2Wider / Net2Deeper")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(
    curves: dict[str, list[float]],
    growth_epoch: int,
    title: str,
    out_path: Path,
) -> None:
    """Test-accuracy curves for grown vs scratch."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, ys in curves.items():
        ax.plot(range(1, len(ys) + 1), [100.0 * y for y in ys], label=name)
    ax.axvline(growth_epoch, color="gray", linestyle=":", label="growth")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _maybe_wandb_init(enabled: bool, config: dict[str, Any]) -> Any:
    if not enabled or wandb_lib is None:
        if enabled and wandb_lib is None:
            print("wandb not installed; continuing without W&B")
        return None
    return wandb_lib.init(
        project="net2net",
        group="repro-figures-2026-09",
        config=config,
        resume="allow",
    )


def run_dataset(
    dataset: str,
    *,
    device: torch.device,
    teacher_epochs: int,
    student_epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    data_dir: Path,
    out_dir: Path,
    wandb_run: Any,
) -> dict[str, Any]:
    """Train teacher → wider/deeper students vs scratch-wider; write curves."""
    if dataset == "mnist":
        train_loader, test_loader = _mnist_loaders(data_dir, batch_size, device)
        make_net = MnistNet
        apply_wider = apply_mnist_wider
        apply_deeper = apply_mnist_deeper
        make_scratch = make_scratch_mnist_wider
        sample = torch.rand(16, 1, 28, 28, device=device)
    elif dataset == "cifar10":
        train_loader, test_loader = _cifar_loaders(data_dir, batch_size, device)
        make_net = CifarNet
        apply_wider = apply_cifar_wider
        apply_deeper = apply_cifar_deeper
        make_scratch = make_scratch_cifar_wider
        sample = torch.rand(16, 3, 32, 32, device=device)
    else:
        msg = f"unknown dataset {dataset}"
        raise ValueError(msg)

    print(f"=== {dataset} on {device} ===", flush=True)
    deltas = measure_growth_deltas(make_net, apply_wider, apply_deeper, sample, device)
    print(f"function deltas: {deltas}", flush=True)

    print("training teacher", flush=True)
    teacher = make_net()
    teacher_hist = train_epochs(
        teacher, train_loader, test_loader, device, teacher_epochs, lr, momentum
    )

    print("training net2wider student", flush=True)
    wider_student = copy.deepcopy(teacher)
    apply_wider(wider_student, noise=False)
    wider_hist = train_epochs(
        wider_student, train_loader, test_loader, device, student_epochs, lr, momentum
    )

    print("training net2deeper student", flush=True)
    deeper_student = copy.deepcopy(teacher)
    apply_deeper(deeper_student, noise=False)
    deeper_hist = train_epochs(
        deeper_student, train_loader, test_loader, device, student_epochs, lr, momentum
    )

    print("training scratch wider", flush=True)
    scratch = make_scratch()
    scratch_hist = train_epochs(
        scratch,
        train_loader,
        test_loader,
        device,
        teacher_epochs + student_epochs,
        lr,
        momentum,
    )

    grown_wider = teacher_hist["test_acc"] + wider_hist["test_acc"]
    grown_deeper = teacher_hist["test_acc"] + deeper_hist["test_acc"]
    plot_convergence(
        {
            "net2wider (from teacher)": grown_wider,
            "net2deeper (from teacher)": grown_deeper,
            "scratch wider": scratch_hist["test_acc"],
        },
        growth_epoch=teacher_epochs,
        title=f"{dataset}: grown vs scratch (test acc)",
        out_path=out_dir / f"{dataset}_convergence_grown_vs_scratch.png",
    )

    payload = {
        "dataset": dataset,
        "function_delta": deltas,
        "teacher_final_test_acc": teacher_hist["test_acc"][-1],
        "net2wider_final_test_acc": grown_wider[-1],
        "net2deeper_final_test_acc": grown_deeper[-1],
        "scratch_wider_final_test_acc": scratch_hist["test_acc"][-1],
        "curves": {
            "net2wider": grown_wider,
            "net2deeper": grown_deeper,
            "scratch_wider": scratch_hist["test_acc"],
        },
    }
    if wandb_run is not None:
        wandb_run.log(
            {
                f"{dataset}/fp_wider_max_abs": deltas["wider"]["max_abs"],
                f"{dataset}/fp_deeper_max_abs": deltas["deeper"]["max_abs"],
                f"{dataset}/teacher_test_acc": payload["teacher_final_test_acc"],
                f"{dataset}/net2wider_test_acc": payload["net2wider_final_test_acc"],
                f"{dataset}/net2deeper_test_acc": payload["net2deeper_final_test_acc"],
                f"{dataset}/scratch_wider_test_acc": payload["scratch_wider_final_test_acc"],
            }
        )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Net2Net MNIST/CIFAR repro figures")
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "both"), default="both")
    parser.add_argument("--teacher-epochs", type=int, default=4)
    parser.add_argument("--student-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "examples" / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run FP measurement + grown-vs-scratch training; write two canonical figures."""
    args = parse_args(argv)
    device = select_device(force_cpu=args.cpu)
    seed_everything(args.seed, device)
    sha = git_sha()
    config = {
        "git_sha": sha,
        "dataset": args.dataset,
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "seed": args.seed,
        "device": str(device),
    }
    wandb_run = _maybe_wandb_init(enabled=not args.no_wandb, config=config)
    datasets_to_run = ("mnist", "cifar10") if args.dataset == "both" else (args.dataset,)
    all_deltas: dict[str, dict[str, dict[str, float]]] = {}
    summaries: dict[str, Any] = {"git_sha": sha, "device": str(device), "runs": {}}
    for dataset in datasets_to_run:
        payload = run_dataset(
            dataset,
            device=device,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            wandb_run=wandb_run,
        )
        all_deltas[dataset] = payload["function_delta"]
        summaries["runs"][dataset] = {
            k: v for k, v in payload.items() if k != "curves"
        } | {"curves": payload["curves"]}
    plot_function_preservation(
        all_deltas, args.out_dir / "function_preservation_error.png"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "repro_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.save(str(args.out_dir / "function_preservation_error.png"))
        for dataset in datasets_to_run:
            wandb_run.save(str(args.out_dir / f"{dataset}_convergence_grown_vs_scratch.png"))
        wandb_run.finish()
    print(json.dumps({k: summaries["runs"][k]["function_delta"] for k in summaries["runs"]}, indent=2))


if __name__ == "__main__":
    main()
