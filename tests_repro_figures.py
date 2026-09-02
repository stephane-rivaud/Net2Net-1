"""Unit tests for canonical Net2Net repro figure helpers (no dataset download)."""

from __future__ import annotations

import torch

from device_utils import select_device
from repro_figures import MnistNet, apply_mnist_deeper, apply_mnist_wider, function_delta


device = select_device()


def test_wider_function_delta_is_small_without_noise() -> None:
    torch.manual_seed(0)
    net = MnistNet().to(device).eval()
    x = torch.rand(8, 1, 28, 28, device=device)
    with torch.no_grad():
        before = net(x)
    apply_mnist_wider(net, noise=False)
    net.eval()
    with torch.no_grad():
        after = net(x)
    delta = function_delta(before, after)
    assert delta["max_abs"] < 1e-4


def test_deeper_function_delta_is_small_without_noise() -> None:
    torch.manual_seed(0)
    net = MnistNet().to(device).eval()
    x = torch.rand(8, 1, 28, 28, device=device)
    with torch.no_grad():
        before = net(x)
    apply_mnist_deeper(net, noise=False)
    net.eval()
    with torch.no_grad():
        after = net(x)
    delta = function_delta(before, after)
    assert delta["max_abs"] < 1e-3


def test_wider_with_noise_breaks_function_preservation() -> None:
    torch.manual_seed(0)
    net = MnistNet().to(device).eval()
    x = torch.rand(8, 1, 28, 28, device=device)
    with torch.no_grad():
        before = net(x)
    apply_mnist_wider(net, noise=True)
    net.eval()
    with torch.no_grad():
        after = net(x)
    delta = function_delta(before, after)
    assert delta["max_abs"] > 1e-4
