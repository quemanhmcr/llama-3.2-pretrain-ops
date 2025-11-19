import torch
import torch.nn as nn
from .args import ModelArgs
from .layers import TransformerBlock, RMSNorm
from .utils import precompute_freqs_cis

class Llama3_2_1B(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, args) for layer_id in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len, self.args.rope_theta, device='cpu')
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)
        for layer in self.layers: h = layer(h, freqs_cis)
        h = self.norm(h)
        return self.output(h)
