import torch
from torch.nn import Module, Parameter, ModuleList
from cs336_basics.attention import scaled_dot_product_attention

class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3.0, b=3.0)
    def forward(self, x):
        return torch.einsum("...i, oi -> ...o", x, self.W)

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x_fp32 / rms
        return (self.gain * x_normed).to(orig_dtype)
    
class SwiGLU(Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x):
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate) 
        itermediate = gate * self.w3(x)
        return self.w2(itermediate)
    
class RoPE(Module):
    def __init__(self, d_k, theta=10000.0, device=None):
        super().__init__()
        self.d_k = d_k
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float().to(device) / d_k))
        self.register_buffer("freqs", freqs)
    def forward(self, x, token_positions):
        angles = token_positions.unsqueeze(-1) * self.freqs
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
        x_rotated = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
        return x * cos + x_rotated * sin
    
class MultiHeadSelfAttention(Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    def forward(self, x, mask=None, rope_layer=None, token_positions=None):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        if rope_layer is not None and token_positions is not None:
            q = rope_layer(q, token_positions)
            k = rope_layer(k, token_positions)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.output_proj(attn_out)
    
class TransformerBlock(Module):
    def __init__(self, d_model, num_heads, d_ff, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x, mask=None, rope_layer=None, token_positions=None):
        x = x + self.attn(self.ln1(x), mask=mask, rope_layer=rope_layer, token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=10000.0, device=None, dtype=None):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RoPE(d_model // num_heads, theta=rope_theta, device=device)
        self.layers = ModuleList([TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    def forward(self, x):
        batch, seq_len = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.view(1, 1, seq_len, seq_len)
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        h = self.token_embeddings(x)
        for layer in self.layers:
            h = layer(h, mask=mask, rope_layer=self.rope, token_positions=token_positions)
        h = self.ln_final(h)
        return self.lm_head(h)
    
def cross_entropy(logits, targets):
    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_stable = logits - logits_max
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
    target_logits = logits_stable[torch.arange(targets.size(0)), targets]
    loss = -target_logits + log_sum_exp
    return torch.mean(loss)

def clip_gradient_norm(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff < 1.0:
        for g in grads:
            g.detach().mul_(clip_coeff)