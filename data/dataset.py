import torch
import random
from typing import Iterator, Dict, List
from torch.utils.data import IterableDataset, Dataset
from .filtering import BigTechDataFilter, FilterStats

class FilteredDatasetView:
    """Filtered view of HuggingFace Dataset - BigTech style."""
    def __init__(self, original_dataset, fixed_keywords: List[str] = None, regex_patterns: List[str] = None,
                 similarity_threshold: float = 0.85, min_text_length: int = 50, num_workers: int = 4,
                 text_field: str = 'text', show_progress: bool = True):
        self.original_dataset = original_dataset
        self.filter_pipeline = BigTechDataFilter(
            sensitive_keywords=fixed_keywords or [],
            similarity_threshold=similarity_threshold,
            min_text_length=min_text_length,
            num_workers=num_workers
        )
        self._kept_indices, self._stats = self.filter_pipeline.filter_dataset(dataset=original_dataset, text_field=text_field)
    
    def __len__(self) -> int: return len(self._kept_indices)
    def __getitem__(self, idx):
        if isinstance(idx, int): return self.original_dataset[self._kept_indices[idx]]
        elif isinstance(idx, slice): return [self.original_dataset[i] for i in self._kept_indices[idx]]
    def __iter__(self):
        for idx in self._kept_indices: yield self.original_dataset[idx]
    def select(self, indices):
        # Hỗ trợ phương thức select giả lập HF Dataset
        return [self[i] for i in indices]

class PretrainDataset(IterableDataset):
    def __init__(self, view, tokenizer, max_length: int = 2048, shuffle_buffer_size: int = 1000, seed: int = 42, drop_last_tokens: bool = True):
        super().__init__()
        self.view = view
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.drop_last_tokens = drop_last_tokens
    
    def _get_worker_info(self) -> tuple:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: return 0, 1, 0, len(self.view), 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            return worker_id, num_workers, worker_id, len(self.view), num_workers
    
    def _shuffle_iterator(self, iterator: Iterator[int], buffer_size: int) -> Iterator[int]:
        buffer = []
        try:
            for _ in range(buffer_size): buffer.append(next(iterator))
        except StopIteration: pass
        while buffer:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            try:
                buffer[idx] = next(iterator)
            except StopIteration:
                buffer.pop(idx)
    
    def _process_text_stream(self, indices: Iterator[int]) -> Iterator[Dict[str, torch.Tensor]]:
        token_buffer = []
        for idx in indices:
            try:
                item = self.view[idx]
                text = item.get('text', '')
                if not text or not text.strip(): continue
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                token_buffer.extend(token_ids)
                token_buffer.append(self.tokenizer.eos_token_id)
                while len(token_buffer) >= self.max_length:
                    chunk = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": input_ids, "labels": input_ids.clone()}
            except Exception: continue
        if not self.drop_last_tokens and len(token_buffer) > 0:
             # Optional: Padding logic usually goes here if keeping last tokens
             pass

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_id, num_workers, iter_start, iter_end, step = self._get_worker_info()
        random.seed(self.seed + worker_id)
        indices = range(iter_start, iter_end, step)
        index_iterator = iter(indices)
        if self.shuffle_buffer_size > 0:
            index_iterator = self._shuffle_iterator(index_iterator, self.shuffle_buffer_size)
        yield from self._process_text_stream(index_iterator)
    
    def __len__(self) -> int:
        avg_tokens_per_doc = 500
        total_tokens = len(self.view) * avg_tokens_per_doc
        return total_tokens // self.max_length

class FintuneDataset(Dataset):
    def __init__(self, view, tokenizer, max_length=1024):
        self.view = view
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.USER_TOKEN = "<|user|>"
        self.ASSISTANT_TOKEN  = "<|assistant|>"

    def __len__(self): return len(self.view)
    
    def __getitem__(self, index):
        conversation = self.view[index]['messages']
        full_input_ids = []
        full_labels = []
        for turn in conversation:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                fomatted_turn = f"{self.USER_TOKEN}\n{content}\n{self.ASSISTANT_TOKEN}\n"
                turn_input_ids = self.tokenizer.encode(fomatted_turn, add_special_tokens=False)
                turn_labels = [-100] * len(turn_input_ids)
            elif role == 'assistant':
                fomatted_turn = f"{content}{self.tokenizer.eos_token}"
                turn_input_ids = self.tokenizer.encode(fomatted_turn, add_special_tokens=False)
                turn_labels = list(turn_input_ids)
            else: continue
            full_input_ids.extend(turn_input_ids)
            full_labels.extend(turn_labels)
        full_input_ids = full_input_ids[:self.max_length]
        full_labels = full_labels[:self.max_length]
        return {"input_ids": torch.tensor(full_input_ids, dtype=torch.long), "labels": torch.tensor(full_labels, dtype=torch.long)}

def create_lazy_dataset(mode, **kwargs):
    if mode == 'pretrain':
        return PretrainDataset(view=kwargs['filtered_view'], tokenizer=kwargs['tokenizer'], max_length=kwargs['max_length'])    
    elif mode == 'finetune':
        return FintuneDataset(filtered_view=kwargs['filtered_view'], tokenizer=kwargs['tokenizer'], max_length=kwargs['max_length'])
    else: raise ValueError(f"Mode không hợp lệ: '{mode}'")
