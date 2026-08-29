import torch
from torch import nn
from torch.nn import functional as F



class SwiGLU(nn.Module):
    r"""Applies the SwiGLU function element-wise.

    SwiGLU is defined as:

    .. math::
        \text{SwiGLU}(x) = a \otimes \text{Swish}(b)

    where :math:`x` is split into :math:`a` and :math:`b` along the specified dimension,
    and :math:`\text{Swish}(b) = b \cdot \sigma(b)`.

    Args:
        dim (int): the dimension along which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = nn.SwiGLU()
        >>> input = torch.randn(4, 2)
        >>> output = m(input)
    """
    __constants__ = ['dim']
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(input, 2, dim=self.dim)
        return a * F.silu(b)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'






class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE).

    Precomputes the inverse frequencies and lazily caches the cos/sin tables for
    the longest sequence seen so far. The rotation is applied per head over the
    ``d_head`` feature dimension, which must be even.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        # inv_freq[i] = base^(-2i / dim) for i in [0, dim/2).
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor = None
        self._sin_cached: torch.Tensor = None

    def _update_cache(self, seq_len: int, device, dtype):
        if (self._cos_cached is None or seq_len > self._seq_len_cached
                or self._cos_cached.device != device or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)           # [seq_len, dim/2]
            emb = torch.cat([freqs, freqs], dim=-1)         # [seq_len, dim]
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, seq_len: int, device, dtype):
        self._update_cache(seq_len, device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the feature dim by swapping/negating its two halves: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to query and key tensors.

    Args:
        q, k: ``[batch, head, seq_len, d_head]``.
        cos, sin: ``[seq_len, d_head]`` rotation tables.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)                     # [1, 1, seq_len, d_head]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MHAttention(nn.Module):
    def __init__(self, d_head, q_head, kv_head=8, dropout_r=0.1, rope_base=10000.0):
        super().__init__()
        self.d_head = d_head
        self.d_q = d_head * q_head
        self.d_kv = d_head * kv_head
        self.q_head = q_head
        self.kv_head = kv_head


        self.wq = nn.Linear(self.d_q, self.d_q)
        self.wk = nn.Linear(self.d_q, self.d_kv)
        self.wv = nn.Linear(self.d_q, self.d_kv)

        self.fc = nn.Linear(self.d_q, self.d_q)

        self.dropout = nn.Dropout(dropout_r)
        self.rms = nn.RMSNorm(self.d_q)
        # RoPE rotates each head's d_head-dim q/k vectors by their token position.
        self.rotary = RotaryEmbedding(d_head, base=rope_base)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # GQA: each kv head is shared by (q_head // kv_head) query heads.
        # Repeat kv heads so the head dim matches q before the matmul.
        n_rep = self.q_head // self.kv_head
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        score = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        output = attn @ v
        return output
    
    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x_q.size()

        q = self.wq(x_q).view(batch_size, seq_len, self.q_head, self.d_head).transpose(1, 2)
        k = self.wk(x_k).view(batch_size, seq_len, self.kv_head, self.d_head).transpose(1, 2)
        v = self.wv(x_v).view(batch_size, seq_len, self.kv_head, self.d_head).transpose(1, 2)

        # Inject positional information by rotating q/k before the attention scores.
        cos, sin = self.rotary(seq_len, x_q.device, q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        output = self.attention(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_q)
        output = self.fc(output)
        output = self.rms(output + x_q)  # Residual connection and normalization
        return output



class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_expert, top_k=2, dropout_r=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout_r)

        self.num_expert = num_expert
        self.gate = nn.Linear(self.d_model, num_expert)
        # Kaiming-style init so experts don't all start saturated (torch.rand is [0, 1)).
        self.ffw1 = nn.Parameter(torch.randn(num_expert, d_model, d_ff) * (d_model ** -0.5))
        self.ffw2 = nn.Parameter(torch.randn(num_expert, d_ff, d_model) * (d_ff ** -0.5))
        self.rms = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x: torch.Tensor):
        residual = x
        B, S, D = x.shape

        logits = self.gate(x)                                   # [B, S, num_expert]

        top_logits, top_e = torch.topk(logits, self.top_k)      # [B, S, top_k]
        # Normalize the gate weights over the selected experts only.
        top_weights = F.softmax(top_logits, dim=-1)             # [B, S, top_k]

        # Flatten the token dimension so routing is a simple gather/scatter over N tokens.
        x_flat = x.reshape(-1, D)                               # [N, D]
        top_e = top_e.reshape(-1, self.top_k)                  # [N, top_k]
        top_weights = top_weights.reshape(-1, self.top_k)      # [N, top_k]

        out = torch.zeros_like(x_flat)                          # [N, D]

        for e in range(self.num_expert):
            # (token, slot) pairs that routed to expert e in any of their top_k slots.
            token_idx, slot_idx = torch.where(top_e == e)
            if token_idx.numel() == 0:
                continue

            x_e = x_flat[token_idx]                             # [n_e, D]
            h = F.silu(x_e @ self.ffw1[e])                      # [n_e, d_ff]
            y = h @ self.ffw2[e]                                # [n_e, D]
            y = self.dropout(y)

            # Scale each expert output by its gate weight, then scatter-add back.
            w = top_weights[token_idx, slot_idx].unsqueeze(-1)  # [n_e, 1]
            out.index_add_(0, token_idx, w * y)

        out = out.reshape(B, S, D)
        return self.rms(out + residual)                         # residual + normalization




class MoEParallel(nn.Module):
    """Drop-in replacement for ``MoE`` that removes the per-expert Python loop.

    The original ``MoE.forward`` iterates over experts one at a time, launching a
    separate pair of matmuls per expert. That serializes the work on the GPU and
    pays Python/CUDA launch overhead ``num_expert`` times per layer.

    Because the expert weights are already stored as stacked tensors
    (``ffw1 [E, D, F]`` / ``ffw2 [E, F, D]``), every expert can be evaluated in a
    single batched matmul (``torch.bmm``). We run all experts densely over every
    token, then combine each token's outputs with a sparse ``[N, E]`` gate-weight
    matrix that is zero for the experts a token did not route to. This is the
    standard "dense" MoE formulation: it is fully parallel and identical in value
    to the loop (in eval mode), at the cost of computing ``num_expert / top_k``x
    more FLOPs. For small/moderate ``num_expert`` the batched GEMM wins easily
    because GPUs are throughput-bound, not FLOP-bound, at these sizes.
    """

    def __init__(self, d_model, d_ff, num_expert, top_k=2, dropout_r=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_expert = num_expert

        self.gate = nn.Linear(self.d_model, num_expert)
        self.ffw1 = nn.Parameter(torch.randn(num_expert, d_model, d_ff) * (d_model ** -0.5))
        self.ffw2 = nn.Parameter(torch.randn(num_expert, d_ff, d_model) * (d_ff ** -0.5))
        self.rms = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x: torch.Tensor):
        residual = x
        B, S, D = x.shape
        E = self.num_expert

        logits = self.gate(x)                                   # [B, S, E]
        top_logits, top_e = torch.topk(logits, self.top_k)      # [B, S, top_k]
        top_weights = F.softmax(top_logits, dim=-1)             # [B, S, top_k]

        x_flat = x.reshape(-1, D)                               # [N, D]
        N = x_flat.shape[0]

        # Build a dense [N, E] gate matrix: combine[n, e] is the gate weight if
        # token n routed to expert e, else 0. Non-selected experts contribute
        # nothing once multiplied in, recovering exact top_k sparsity in value.
        combine = x_flat.new_zeros(N, E)
        combine.scatter_(1, top_e.reshape(N, self.top_k), top_weights.reshape(N, self.top_k))

        # All experts process all tokens in one batched GEMM. expand() is a view
        # (no copy); bmm broadcasts over the expert/batch dimension.
        xb = x_flat.unsqueeze(0).expand(E, N, D)               # [E, N, D]
        h = F.silu(torch.bmm(xb, self.ffw1))                   # [E, N, d_ff]
        y = torch.bmm(h, self.ffw2)                            # [E, N, D]

        # Weighted sum over experts: out[n] = sum_e combine[n, e] * y[e, n].
        out = torch.einsum('ne,end->nd', combine, y)           # [N, D]
        out = self.dropout(out)

        out = out.reshape(B, S, D)
        return self.rms(out + residual)




class MoEGrouped(nn.Module):
    """Loop-free MoE that keeps sparse compute and scales to many experts.

    ``MoEParallel`` removes the Python loop but evaluates every expert on every
    token, so its FLOPs grow with ``num_expert`` and it only wins when the expert
    count is small. This class instead does what production MoE kernels do:

      1. Expand each token's ``top_k`` routing decisions into a flat list of
         (token, expert) assignments  ->  ``N * top_k`` rows.
      2. Sort the rows by expert id so every expert's tokens are contiguous.
      3. Run all experts in ONE batched grouped matmul (``torch._grouped_mm``),
         where ``offs`` marks each expert's slice. No Python loop, no wasted FLOPs
         (each token is computed exactly ``top_k`` times, same as the loop).
      4. Scatter the weighted results back to their original token positions.

    ``torch._grouped_mm`` runs in bf16/fp16 (it targets tensor cores), so the
    expert matmuls here are bf16 while the gate, combine, and norm stay fp32 --
    the usual mixed-precision split. Output therefore matches ``MoE`` to bf16
    tolerance, not bit-for-bit. Requires a recent PyTorch + a tensor-core GPU.
    """

    def __init__(self, d_model, d_ff, num_expert, top_k=2, dropout_r=0.1, compute_dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_expert = num_expert
        self.compute_dtype = compute_dtype

        self.gate = nn.Linear(self.d_model, num_expert)
        self.ffw1 = nn.Parameter(torch.randn(num_expert, d_model, d_ff) * (d_model ** -0.5))
        self.ffw2 = nn.Parameter(torch.randn(num_expert, d_ff, d_model) * (d_ff ** -0.5))
        self.rms = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x: torch.Tensor):
        residual = x
        B, S, D = x.shape
        E, k = self.num_expert, self.top_k

        logits = self.gate(x)                                   # [B, S, E]
        top_logits, top_e = torch.topk(logits, k)               # [B, S, k]
        top_weights = F.softmax(top_logits, dim=-1)             # [B, S, k]

        x_flat = x.reshape(-1, D)                               # [N, D]
        N = x_flat.shape[0]

        # One row per (token, selected-expert) assignment: M = N * k rows.
        flat_expert = top_e.reshape(-1)                         # [M]
        flat_weight = top_weights.reshape(-1)                   # [M]
        token_id = torch.arange(N, device=x.device).repeat_interleave(k)  # [M]

        # Sort assignments by expert so each expert owns a contiguous slice.
        order = torch.argsort(flat_expert)
        sorted_expert = flat_expert[order]
        sorted_token = token_id[order]
        sorted_weight = flat_weight[order]

        # offs[e] = end index of expert e's slice (cumulative token counts).
        counts = torch.bincount(sorted_expert, minlength=E)
        offs = counts.cumsum(0).to(torch.int32)                 # [E]

        cdt = self.compute_dtype
        xg = x_flat[sorted_token].to(cdt)                       # [M, D]
        h = torch._grouped_mm(xg, self.ffw1.to(cdt), offs=offs) # [M, d_ff]
        h = F.silu(h)
        yg = torch._grouped_mm(h, self.ffw2.to(cdt), offs=offs) # [M, D]

        # Weight each expert output, then scatter-add back to its source token.
        yg = yg * sorted_weight.unsqueeze(-1)                   # [M, D], bf16
        out = x_flat.new_zeros(N, D)                            # [N, D], bf16
        out.index_add_(0, sorted_token, yg)
        out = self.dropout(out)

        out = out.reshape(B, S, D)
        return self.rms(out + residual)


class TransformerBlock(nn.Module):
    def __init__(self, d_head, q_head, kv_head, num_experts, top_k, dropout_r=0.1):
        super().__init__()
        self.mh_attn = MHAttention(d_head=d_head, q_head=q_head, kv_head=kv_head, dropout_r=dropout_r)
        self.moe_ffn = MoEGrouped(d_model=d_head*q_head, d_ff=d_head*q_head*8, num_expert=num_experts, top_k=top_k, dropout_r=dropout_r, compute_dtype=torch.bfloat16)

    
    def forward(self, x: torch.Tensor):
        x = self.mh_attn(x_q=x, x_k=x, x_v=x)
        x = self.moe_ffn(x)

        return x






class Brainer_MoE(nn.Module):
    def __init__(self, d_head, q_head, kv_head, num_experts, top_k, num_layers, vocab_size, dropout_r=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_head*q_head, dtype=torch.bfloat16)
        self.transformer = nn.ModuleList([
            TransformerBlock(d_head=d_head, q_head=q_head, kv_head=kv_head,
                             num_experts=num_experts, top_k=top_k, dropout_r=dropout_r)
            for _ in range(num_layers)
        ])
        self.post_norm = nn.RMSNorm(d_head*q_head)
        self.out = nn.Linear(d_head*q_head, vocab_size)

    
    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for layer in self.transformer:
            x = layer(x)
        x = self.out(self.post_norm(x))
        return x




if __name__ == "__main__":
    # Example usage
    d_head = 64
    q_head = 16
    kv_head = 8

    num_experts = 4
    top_k = 2

    num_layers = 48

    dropout_r = 0.1

    gemma4_vocab_size = 262144


    seq_len = 10
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mhattention = MHAttention(d_head, q_head, kv_head).to(device)
    

    # x = torch.rand(batch_size, seq_len, d_head * q_head).to(device)
    # print(f"Input shape: {x.shape}")  # Should be (batch_size, seq_len, d_head * q_head)
    # mask = torch.ones(batch_size, seq_len, seq_len).to(device)  # Example mask

    # mhattention_out = mhattention(x_q=x, x_k=x, x_v=x, mask=mask)
    # print(f"mhattention_out shape: {mhattention_out.shape}")  # Should be (batch_size, seq_len, d_head * q_head)

    # moe = MoEGrouped(d_model=q_head*d_head, d_ff=q_head*d_head*2, num_expert=4, top_k=2, dropout_r=0.1, compute_dtype=torch.bfloat16).to(device)
    # moe_out = moe(mhattention_out)
    # print(f"moe_out shape: {moe_out.shape}")

    # transformer_block = TransformerBlock(d_head=d_head, q_head=q_head, kv_head=kv_head, num_experts=num_experts, top_k=top_k, dropout_r=dropout_r).to(device)
    # transformer_block_out = transformer_block(x)
    # print(f"transformer_block_out shape: {transformer_block_out.shape}")

    with torch.no_grad():
        x = torch.randint(low=0, high=gemma4_vocab_size, size=(batch_size, seq_len)).to(device)
        # Cast to bf16 BEFORE moving to device: `.to(device)` first would
        # materialize all 2.79B params in fp32 (~11GB) on the GPU before the
        # cast, sitting at the edge of a 12GB card and causing flaky
        # CUBLAS_STATUS_ALLOC_FAILED. This order moves only ~5.6GB of bf16.
        model = Brainer_MoE(d_head=d_head, q_head=q_head, kv_head=kv_head, num_experts=num_experts, top_k=top_k, num_layers=num_layers, vocab_size=gemma4_vocab_size, dropout_r=dropout_r).to(torch.bfloat16).to(device)

        import torchinfo
        torchinfo.summary(model=model, input_data=x, device=device)

        model_out = model(x)
        print(f"Bainer_MoE model output shape: {model_out.shape}")