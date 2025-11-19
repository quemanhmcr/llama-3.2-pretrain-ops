from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128000
    ffn_hidden_dim: int = 8192
    max_seq_len: int = 131072
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
