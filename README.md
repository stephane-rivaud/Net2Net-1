# Net2Net
Net2Net implementation on PyTorch for any possible vision layers (nn.Linear, nn.Conv2d, nn.Conv3D, even wider operator btw nn.ConvX to nn.Linear).
Checkout the [paper](https://arxiv.org/abs/1511.05641:) for more detail

## Quickstart

Create a clean project environment with `uv` and run the test suite:

```bash
uv venv
uv sync
uv run pytest -q tests.py
```

Or use `just`:

```bash
just venv
just sync
just test
```

## Repro figures (MNIST / CIFAR proxies)

Canonical plots for function preservation at growth and grown-vs-scratch
convergence (W&B project `net2net`, group `repro-figures-2026-09`):

```bash
uv run python repro_figures.py --dataset both --teacher-epochs 4 --student-epochs 4
```

Outputs land in `figures/function_preservation_error.png` and
`figures/{mnist,cifar10}_convergence_grown_vs_scratch.png`.
Operator-level tests (no data download): `uv run pytest -q tests.py tests_repro_figures.py`.

On the 2026-09-03 MPS proxy (`--teacher-epochs 4 --student-epochs 4`):

- MNIST: function preservation holds (`wider` max |Δ| ≈ 2e-7, `deeper` ≈ 0).
  Final test acc: teacher 98.21%, net2wider 98.81%, net2deeper 98.89%,
  scratch-wider 98.91% (ceiling; no speed advantage).
- CIFAR-10: `deeper` preserves function (max |Δ| ≈ 4e-6) and finishes at
  71.93% vs scratch-wider 69.13%. `wider` does **not** preserve function in
  this BN-in-forward wrapping (max |Δ| ≈ 0.16) and finishes at 63.81%.

W&B: project [`net2net`](https://wandb.ai/TAU-Frugal/net2net), group
`repro-figures-2026-09`.

If you need to reset the environment, remove `.venv` and rerun `uv venv` and `uv sync`.

## Observations:

- Using BatchNorm between layers, improves the competence of Net2Net. Otherwise, Net2Net approach is not able to get
comparable results to a network trained from scratch.

- Inducing noise to new units and connections prelude to better networks. The effect is more evident without BathNorm layer.

- Normalizing layer weights before any Net2Net operation increases the speed of learning and gives better convergence. Even so, it worths to investgate better normalization methods than L2 norm.
