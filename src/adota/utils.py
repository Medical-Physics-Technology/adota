import torch
from torch.nn import functional as F
from .models import DoTA3D_v3
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def count_parameters_per_block(model: torch.nn.Module):
    from collections import defaultdict
    import torch.nn as nn

    block_params = defaultdict(int)

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            block_params[name] += sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

    for block, count in block_params.items():
        print(f"{block}: {count:,} parameters")


def count_total_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params:,}")
    return total_params


def load_model(
    model_path: Path,
    hyperparams_path: Path,
    device: torch.device,
) -> DoTA3D_v3:
    """Load and configure the DoTA model.

    Args:
        model_path: Path to the model weights file.
        hyperparams_path: Path to the hyperparameters JSON file.
        device: Target device for the model.

    Returns:
        Configured DoTA3D_v3 model in eval mode.

    Raises:
        FileNotFoundError: If model or hyperparams files don't exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not hyperparams_path.exists():
        raise FileNotFoundError(f"Hyperparams file not found: {hyperparams_path}")

    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)

    model = DoTA3D_v3(**hyperparams)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    logger.info(f"Model loaded from {model_path}")
    return model
