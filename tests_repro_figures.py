"""Unit tests for canonical Net2Net repro figure helpers (no dataset download)."""

from __future__ import annotations

import torch

import torch.nn as nn
import torch.nn.functional as F

from device_utils import select_device
from net2net import wider
from repro_figures import (
    CifarNet,
    MnistNet,
    apply_cifar_wider,
    apply_mnist_deeper,
    apply_mnist_wider,
    function_delta,
)


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


def test_wider_bn_conv_block_function_preservation() -> None:
    """Net2Wider on a BN conv block must keep eval-mode outputs (≤ 1e-5)."""
    torch.manual_seed(0)
    conv1 = nn.Conv2d(3, 8, 3, padding=1).to(device)
    bn1 = nn.BatchNorm2d(8).to(device)
    conv2 = nn.Conv2d(8, 16, 3, padding=1).to(device)
    with torch.no_grad():
        bn1.running_mean.uniform_(-0.5, 0.5)
        bn1.running_var.uniform_(0.4, 1.6)
        bn1.weight.uniform_(0.7, 1.3)
        bn1.bias.uniform_(-0.2, 0.2)
    x = torch.rand(4, 3, 16, 16, device=device)
    conv1.eval()
    bn1.eval()
    conv2.eval()
    with torch.no_grad():
        before = conv2(F.relu(bn1(conv1(x))))
    wider(
        conv1,
        conv2,
        12,
        bn1,
        noise=False,
        random_init=False,
        weight_norm=False,
    )
    conv1.eval()
    bn1.eval()
    conv2.eval()
    with torch.no_grad():
        after = conv2(F.relu(bn1(conv1(x))))
    delta = function_delta(before, after)
    assert delta["max_abs"] <= 1e-5, delta


def test_cifar_wider_function_delta_is_small_without_noise() -> None:
    torch.manual_seed(0)
    net = CifarNet().to(device).eval()
    x = torch.rand(8, 3, 32, 32, device=device)
    with torch.no_grad():
        before = net(x)
    apply_cifar_wider(net, noise=False)
    net.eval()
    with torch.no_grad():
        after = net(x)
    delta = function_delta(before, after)
    assert delta["max_abs"] <= 1e-5, delta
