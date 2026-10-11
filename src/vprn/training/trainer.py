from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


class Trainer:
    """Small, reusable training loop; case-specific behavior is supplied by step_fn."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
        scheduler: Any | None = None,
        device: str | torch.device = "cpu",
        accumulate_steps: int = 1,
        gradient_clip_norm: float | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.step_fn = step_fn
        self.scheduler = scheduler
        self.device = torch.device(device)
        if accumulate_steps <= 0:
            raise ValueError("accumulate_steps must be positive")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        self.accumulate_steps = int(accumulate_steps)
        self.gradient_clip_norm = gradient_clip_norm

    def fit(
        self,
        batches: Iterable[Any],
        epochs: int,
        *,
        progress: bool = False,
        on_epoch_end: Callable[[int, dict[str, float | int], list[dict[str, float | int]]], None]
        | None = None,
    ) -> list[dict[str, float | int]]:
        """Train for ``epochs`` and optionally show a per-batch tqdm progress bar."""
        materialized = list(batches)
        if epochs < 0:
            raise ValueError("epochs must be non-negative")
        history: list[dict[str, float | int]] = []
        for epoch in range(epochs):
            self.model.train()
            losses = []
            self.optimizer.zero_grad(set_to_none=True)
            progress_bar = None
            batch_iterator = materialized
            if progress:
                from tqdm.auto import tqdm

                progress_bar = tqdm(
                    materialized,
                    total=len(materialized),
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                )
                batch_iterator = progress_bar
            for batch_index, batch in enumerate(batch_iterator):
                loss = self.step_fn(self.model, batch)
                if loss.ndim != 0:
                    loss = loss.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}: {loss.item()}")
                loss.backward()
                should_step = (batch_index + 1) % self.accumulate_steps == 0 or batch_index == len(materialized) - 1
                if should_step:
                    if self.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        loss=f"{loss_value:.5f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    )
            if progress_bar is not None:
                progress_bar.close()
            record = {"epoch": epoch, "loss": sum(losses) / len(losses) if losses else float("nan")}
            history.append(record)
            if on_epoch_end is not None:
                on_epoch_end(epoch, record, history)
        return history
