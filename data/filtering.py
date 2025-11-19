import ahocorasick
import multiprocessing
import logging
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
from .deduplication import MinHashDeduplicator

@dataclass
class FilterStats:
    total_docs: int = 0
    sensitive_filtered: int = 0
    duplicates_filtered: int = 0
    empty_filtered: int = 0
    kept: int = 0
    processing_time: float = 0.0

class SensitiveContentFilter:
    """Fast sensitive content detection using Aho-Corasick only."""
    __slots__ = ('automaton', 'keyword_count')
    
    def __init__(self, keywords: Iterable[str]):
        self.automaton = ahocorasick.Automaton()
        unique_keywords = {kw.lower() for kw in keywords if kw}
        self.keyword_count = len(unique_keywords)
        for idx, keyword in enumerate(unique_keywords):
            self.automaton.add_word(keyword, (idx, keyword))
        self.automaton.make_automaton()
    
    def is_sensitive(self, text: str) -> bool:
        if not text: return False
        return next(self.automaton.iter(text.lower()), None) is not None

def _process_shard_worker(args):
    (shard_id, start_idx, end_idx, dataset, text_field,
     keywords, similarity_threshold, min_text_length) = args
    
    logging.info(f"Shard {shard_id}: Starting processing [{start_idx:,} - {end_idx:,}]")
    sensitive_filter = SensitiveContentFilter(keywords) if keywords else None
    deduplicator = MinHashDeduplicator(threshold=similarity_threshold)
    stats = FilterStats()
    kept_indices = []
    
    for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
        stats.total_docs += 1
        try:
            doc = dataset[global_idx]
            text = doc.get(text_field, '') if isinstance(doc, dict) else getattr(doc, text_field, '')
        except Exception as e:
            stats.empty_filtered += 1
            continue
        
        if not text or len(text) < min_text_length:
            stats.empty_filtered += 1
            continue
        
        if sensitive_filter and sensitive_filter.is_sensitive(text):
            stats.sensitive_filtered += 1
            continue
        
        doc_id = f"shard{shard_id}_doc{global_idx}"
        minhash = deduplicator.create_minhash(text)
        if deduplicator.query(minhash):
            stats.duplicates_filtered += 1
            continue
        
        deduplicator.insert(doc_id, minhash)
        kept_indices.append(global_idx)
        stats.kept += 1
    
    return kept_indices, stats

class BigTechDataFilter:
    def __init__(self, sensitive_keywords: List[str], similarity_threshold: float = 0.8, min_text_length: int = 50, num_workers: int = 4):
        self.min_text_length = min_text_length
        self.num_workers = num_workers
        self.sensitive_keywords = sensitive_keywords
        self.sensitive_filter = SensitiveContentFilter(sensitive_keywords)
        self.similarity_threshold = similarity_threshold

    def _create_shards(self, dataset, num_shards: Optional[int] = None) -> List[Tuple[int, int, int]]:
        if num_shards is None: num_shards = self.num_workers
        total_size = len(dataset)
        shard_size = total_size // num_shards
        shards = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < num_shards - 1 else total_size
            shards.append((i, start_idx, end_idx))
        return shards
    
    def filter_dataset(self, dataset, text_field: str = 'text'):
        shards = self._create_shards(dataset)
        worker_args = [
            (shard_id, start_idx, end_idx, dataset, text_field,
             self.sensitive_keywords, self.similarity_threshold, self.min_text_length)
            for shard_id, start_idx, end_idx in shards
        ]
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.map(_process_shard_worker, worker_args)
        
        all_kept_indices = []
        total_stats = FilterStats()
        for kept_indices, stats in results:
            all_kept_indices.extend(kept_indices)
            total_stats.total_docs += stats.total_docs
            total_stats.sensitive_filtered += stats.sensitive_filtered
            total_stats.duplicates_filtered += stats.duplicates_filtered
            total_stats.empty_filtered += stats.empty_filtered
            total_stats.kept += stats.kept
        return all_kept_indices, total_stats
