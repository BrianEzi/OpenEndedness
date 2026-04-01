import json
import os

files = [
    "models/fsq.py",
    "models/seer.py",
    "models/doer.py",
    "envs/navix_wrapper.py",
    "agents/mappo.py",
    "training/gae.py",
    "training/loop.py",
    "eval/metrics.py",
    "eval/visualize.py",
    "train.py"
]

cells = []

# Headers and deps
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["# Train JaxMarl Models in Google Colab\n", "Ensure you are using a GPU runtime (`Runtime` -> `Change runtime type` -> `T4 GPU` or similar)."]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Install dependencies\n", "!pip install --upgrade numpy\n", "!pip install jax[cuda12] flax optax wandb\n", "!pip install git+https://github.com/FLAIROx/JaxMARL.git\n", "!pip install navix Pillow matplotlib distrax chex"]})

# Imports
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "import jax\n", "import jax.numpy as jnp\n", "import flax.linen as nn\n", "import optax\n", "import distrax\n", "import chex\n", "import wandb\n", "import navix as nx\n", "import numpy as np\n", "from pathlib import Path\n", "from flax.training.train_state import TrainState\n", "import functools\n", "from typing import Any, Callable, Dict, Tuple, Sequence\n", "from PIL import Image\n"
]})

for f in files:
    with open(f, "r") as r:
        lines = r.readlines()
        
    filtered = []
    for line in lines:
        if line.startswith("import ") or line.startswith("from "):
            if "models." in line or "envs." in line or "training." in line or "agents." in line or "eval." in line:
                continue
            if "train import" in line:
                continue
        filtered.append(line)
        
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": filtered})
    
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["wandb.login()"]})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["main()"]})

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("train.ipynb", "w") as w:
    json.dump(notebook, w, indent=1)
