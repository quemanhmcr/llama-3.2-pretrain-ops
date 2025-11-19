import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: str = 'cpu') -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    COMPLEX_PAIR = 2
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, COMPLEX_PAIR))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, COMPLEX_PAIR))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
