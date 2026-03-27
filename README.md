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

If you need to reset the environment, remove `.venv` and rerun `uv venv` and `uv sync`.

## Observations:

- Using BatchNorm between layers, improves the competence of Net2Net. Otherwise, Net2Net approach is not able to get
comparable results to a network trained from scratch.

- Inducing noise to new units and connections prelude to better networks. The effect is more evident without BathNorm layer.

- Normalizing layer weights before any Net2Net operation increases the speed of learning and gives better convergence. Even so, it worths to investgate better normalization methods than L2 norm.
