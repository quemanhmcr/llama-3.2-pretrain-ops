import torch
import torch.nn.functional as F
import wandb
import os
import traceback
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from configs.training_config import TrainingConfig
import torch.nn as nn
from dataclasses import asdict

class ProductionTrainer:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader], config: TrainingConfig):
        self.model = model.to(config.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.optimizer = self._create_optimizer()
        self.use_amp = config.use_amp and config.device == "cuda"
        self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None
        self.scheduler = self._create_scheduler()
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.steps_since_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self._init_loggers()

    def _create_optimizer(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if "bias" in name or "norm" in name or "ln" in name: no_decay_params.append(param)
            else: decay_params.append(param)
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, betas=(self.config.adam_beta1, self.config.adam_beta2), eps=self.config.adam_epsilon)

    def _create_scheduler(self):
        total_steps = self._get_total_steps()
        return get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=total_steps)

    def _get_total_steps(self):
        if self.config.max_steps: return self.config.max_steps
        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _init_loggers(self):
        if self.config.use_wandb:
            try:
                wandb.init(project=self.config.wandb_project, config=self.config.to_dict())
            except Exception: self.config.use_wandb = False
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(os.path.join(self.config.save_dir, "tensorboard"))
            except Exception: self.config.use_tensorboard = False

    def _log_metrics(self, metrics, step):
        if step % self.config.log_steps == 0:
            print(f"Step {step}: " + " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
        if self.config.use_wandb: wandb.log(metrics, step=step)
        if self.config.use_tensorboard:
            for k, v in metrics.items(): self.writer.add_scalar(k, v, step)

    def _save_checkpoint(self, val_loss=None, is_best=False):
        checkpoint = {
            'epoch': self.epoch, 'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), 'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss
        }
        path = Path(self.config.save_dir) / ("best_model.pt" if is_best else f"checkpoint_{self.global_step}.pt")
        torch.save(checkpoint, path)
        if not is_best: self._cleanup_checkpoints(Path(self.config.save_dir))

    def _cleanup_checkpoints(self, checkpoint_dir):
        ckpts = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=lambda x: int(x.stem.split('_')[1]))
        for c in ckpts[:-self.config.save_total_limit]: c.unlink(missing_ok=True)

    def train_step(self, batch):
        self.model.train()
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(input_ids)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            loss = loss / self.config.gradient_accumulation_steps
        if self.use_amp: self.scaler.scale(loss).backward()
        else: loss.backward()
        return loss.item(), {"loss": loss.item() * self.config.gradient_accumulation_steps, "learning_rate": self.scheduler.get_last_lr()[0]}

    def optimizer_step(self):
        if self.use_amp: self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else: self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm.item()

    @torch.no_grad()
    def evaluate(self, max_eval_batches=None):
        if not self.val_dataloader: return None
        if max_eval_batches is None: max_eval_batches = getattr(self.config, 'num_val_batches', 1000)
        self.model.eval()
        total_loss, batches = 0.0, 0
        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(input_ids)
                loss = F.cross_entropy(outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)), labels[..., 1:].contiguous().view(-1), ignore_index=-100)
            total_loss += loss.item()
            batches += 1
            if batches >= max_eval_batches: break
        return total_loss / batches

    def train(self):
        print("🚀 STARTING TRAINING")
        self.model.train()
        accumulated_loss = 0.0
        step_in_epoch = 0
        try:
            for epoch in range(self.config.num_epochs):
                self.epoch = epoch
                print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
                for batch in self.train_dataloader:
                    loss, metrics = self.train_step(batch)
                    accumulated_loss += loss
                    step_in_epoch += 1
                    if step_in_epoch % self.config.gradient_accumulation_steps == 0:
                        grad_norm = self.optimizer_step()
                        self.global_step += 1
                        metrics["grad_norm"] = grad_norm
                        metrics["loss"] = accumulated_loss
                        self._log_metrics(metrics, self.global_step)
                        accumulated_loss = 0.0
                        if self.global_step % self.config.eval_steps == 0:
                            val_loss = self.evaluate()
                            if val_loss:
                                print(f"Validation loss: {val_loss:.4f}")
                                if val_loss < self.best_val_loss:
                                    self.best_val_loss = val_loss
                                    self.steps_since_improvement = 0
                                    self._save_checkpoint(val_loss, True)
                                else: self.steps_since_improvement += 1
                                if self.steps_since_improvement >= self.config.early_stopping_patience: return
                        if self.global_step % self.config.save_steps == 0: self._save_checkpoint()
                        if self.config.max_steps and self.global_step >= self.config.max_steps: return
                if self.val_dataloader:
                    val_loss = self.evaluate()
                    if val_loss and val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(val_loss, True)
        except KeyboardInterrupt:
            print("Training interrupted")
            self._save_checkpoint()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            self._save_checkpoint()
            raise e
