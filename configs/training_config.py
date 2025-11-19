from dataclasses import dataclass, asdict
from typing import Optional, Dict
from pathlib import Path
import torch
import os

@dataclass
class TrainingConfig:
    model_name: str = "llama3.2-1b-pretrain"
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    lr_scheduler_type: str = "cosine"
    use_amp: bool = True
    amp_dtype: str = "float16"
    save_dir: str = "./checkpoints"
    save_steps: int = 1000
    save_total_limit: int = 3
    log_steps: int = 10
    eval_steps: int = 500
    use_wandb: bool = True
    use_tensorboard: bool = False
    wandb_project: str = "llama-pretraining"
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    debug_mode: bool = False
    profile_memory: bool = False
    
    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
    def to_dict(self) -> Dict: return asdict(self)
