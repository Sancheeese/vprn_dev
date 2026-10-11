from __future__ import annotations

import torch
from torchvision.transforms import Compose, Normalize, Resize


class ImageTransforms:
    """Shared X-ray/DRR normalization used by all configured cases."""

    def __init__(self, size: int = 256, radius: int = 121, eps: float = 1e-6):
        self.eps = eps
        self.resize = Resize((size, size), antialias=True)
        self.normalize = Compose([Normalize(mean=0.3080, std=0.1494)])
        y_coord = torch.arange(size) - size // 2
        x_coord = torch.arange(size) - size // 2
        y_grid, x_grid = torch.meshgrid(y_coord, x_coord, indexing="ij")
        self.mask = ((x_grid * x_grid + y_grid * y_grid) <= radius * radius).float()

    def __call__(self, image: torch.Tensor, *, reverse: bool = True) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            image = image.unsqueeze(0)
        image = self.resize(image.float())
        image = (image - image.amin()) / (image.amax() - image.amin() + self.eps)
        if reverse:
            image = 1.0 - image
        mask = self.mask.to(image.device).view(1, 1, *self.mask.shape)
        return self.normalize(image * mask)
