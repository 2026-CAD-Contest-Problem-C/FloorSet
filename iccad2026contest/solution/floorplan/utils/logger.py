"""W&B / TensorBoard wrapper."""
import os


class Logger:
    """Lightweight logger with optional W&B support."""

    def __init__(self, project: str = 'floorset', enabled: bool = True, **kwargs):
        self.enabled = enabled
        self._wandb = None

        if enabled:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(project=project, **kwargs)
            except ImportError:
                print("wandb not available, logging disabled")
                self.enabled = False

    def log(self, metrics: dict, step: int = None):
        if self._wandb and self.enabled:
            self._wandb.log(metrics, step=step)

    def finish(self):
        if self._wandb and self.enabled:
            self._wandb.finish()
