import os
import torch
import logging
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from configs.training_config import TrainingConfig
from data.processor import OpenWebTextProcessor
from data.dataset import PretrainDataset
from models.args import ModelArgs
from models.modeling_llama import Llama3_2_1B
from training.trainer import ProductionTrainer

# Thiết lập logging cơ bản
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Llama 3.2 1B")
    parser.add_argument("--data_path", type=str, required=True, help="Đường dẫn đến file dữ liệu huấn luyện (txt)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Thư mục lưu checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=3, help="Số lượng epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Độ dài chuỗi tối đa")
    parser.add_argument("--wandb", action="store_true", help="Sử dụng WandB để log")
    
    args = parser.parse_args()

    # --- Update Config ---
    @dataclass
    class TrainingConfigV2(TrainingConfig):
        batch_size: int = args.batch_size
        max_seq_len: int = args.max_seq_len
        gradient_accumulation_steps: int = 8
        num_epochs: int = args.epochs
        max_steps: Optional[int] = None
        learning_rate: float = args.lr
        save_dir: str = args.output_dir
        log_steps: int = 10
        eval_steps: int = 500
        num_val_batches: int = 1000
        use_wandb: bool = args.wandb
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainingConfigV2()
    
    # --- Tokenizer ---
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.eos_token is None: raise ValueError("Tokenizer needs EOS token")
    
    # --- Data Processing ---
    print("\n📊 Creating datasets...")
    file_path = args.data_path
    
    if os.path.exists(file_path):
        owt_dataset = OpenWebTextProcessor(
            file_path=file_path,
            chunk_Size=64 * 1024 * 1024,
            num_workers=os.cpu_count()
        )
        owt_dataset.build_dataset()
        
        if owt_dataset.dataset:
            original_dataset = owt_dataset.dataset
            train_size = int(0.9 * len(original_dataset))
            val_size = len(original_dataset) - train_size
            
            train_view = original_dataset.select(range(train_size))
            val_view = original_dataset.select(range(train_size, len(original_dataset)))
            
            # --- Dataset Objects ---
            train_dataset = PretrainDataset(view=train_view, tokenizer=tokenizer, max_length=config.max_seq_len)
            val_dataset = PretrainDataset(view=val_view, tokenizer=tokenizer, max_length=config.max_seq_len)
            
            # --- DataLoaders ---
            print("\n🔄 Creating dataloaders...")
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory)
            
            # --- Initialize Model ---
            print("\n🤖 Initializing model...")
            model_args = ModelArgs(
                dim=1024, n_layers=12, n_heads=16, n_kv_heads=4, vocab_size=len(tokenizer),
                ffn_hidden_dim=4096, max_seq_len=config.max_seq_len
            )
            model = Llama3_2_1B(model_args)
            
            # --- Cleanup ---
            # !pip install numba
            import gc
            try:
                from numba import cuda
                device = cuda.get_current_device()
                device.reset()
            except: pass
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            # --- Training ---
            print("\n🏋️ Creating trainer...")
            trainer = ProductionTrainer(model=model, train_dataloader=train_loader, val_dataloader=val_loader, config=config)
            trainer.train()
            
            # --- Inference Test ---
            print("\nStarting generation tests...")
            device = config.device
            
            # Reload best model
            try:
                checkpoint = torch.load(f"{config.save_dir}/best_model.pt", map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            except: pass
            
            model.eval()
            
            @torch.no_grad()
            def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=20):
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                for _ in range(max_new_tokens):
                    outputs = model(prompt_ids)
                    logits = outputs[:, -1, :] / temperature
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id: break
                return tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            
            prompts = ["The best way to learn", "Artificial intelligence is"]
            for p in prompts:
                print(f"\nPrompt: {p}")
                print(f"Generated: {generate(p)}")
    else:
        print(f"File input không tồn tại: {file_path}. Vui lòng kiểm tra đường dẫn.")
