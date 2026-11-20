# Installation

NIMBUS needs Python 3.10 or newer. It runs on Linux, macOS and Windows; the GPU backend needs an NVIDIA GPU.

=== "From source (recommended)"

    The repository holds the examples, the experiments and the hardware designs as well as the package:

    ```bash
    git clone https://github.com/aimlab-wustl/Nimbus.git
    cd Nimbus
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -e ".[dev]"
    ```

=== "pip from GitHub"

    The package alone, without the examples and experiments:

    ```bash
    pip install "nimbus-ising @ git+https://github.com/aimlab-wustl/Nimbus.git"
    ```

=== "With the GPU backend"

    From the repository root, with CUDA 12:

    ```bash
    pip install -e ".[dev,gpu]"
    ```

## Extras

| Extra | Installs | Needed for |
|---|---|---|
| `dev` | pytest, matplotlib, pandas | tests, figures, reading results as data frames |
| `plot` | matplotlib, pandas | figures only |
| `refs` | PyMatching, ldpc, stim, qecsim | the CPU reference decoders of the surface-code experiments |
| `gpu` | CuPy for CUDA 12 | the optional GPU backend (NVIDIA GPU) |
| `docs` | MkDocs Material | building this website |

## Check the installation

```bash
nimbus doctor
nimbus problems
pytest -q
```

`nimbus doctor` lists the backends that are available on this machine and the data and runs folders.
`nimbus problems` lists the problem kinds a configuration can name. The first run of a backend compiles its Numba
kernels (cached afterwards), so the first solve takes a few seconds longer than later ones.

Next: [Quickstart](quickstart.md).
