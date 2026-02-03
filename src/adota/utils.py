import torch
from torch.nn import functional as F


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
