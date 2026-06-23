"""Benchmark: looped MoE vs. batched MoEParallel vs. grouped MoEGrouped.

Measures forward (inference) and forward+backward (training) latency, and
checks numerical agreement against the reference loop.

Run: python model/bench_moe.py
"""
import time
import torch
from model import MoE, MoEParallel, MoEGrouped


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def share_weights(ref: MoE, other):
    """Copy ref's params into another MoE variant so only the algo differs."""
    other.gate.load_state_dict(ref.gate.state_dict())
    other.ffw1.data.copy_(ref.ffw1.data)
    other.ffw2.data.copy_(ref.ffw2.data)
    other.rms.load_state_dict(ref.rms.state_dict())


@torch.no_grad()
def rel_err(ref: MoE, other, x):
    ref.eval(); other.eval()
    a, b = ref(x), other(x)
    return ((a - b).norm() / a.norm()).item()


def fmt_err(ref, other, x):
    e = safe(rel_err, ref, other, x)
    return "OOM" if e is None else f"{e:.2e}"


def bench_forward(model, x, device, iters=50, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync(device)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def bench_train(model, x, device, iters=50, warmup=10):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        opt.step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        opt.step()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def safe(fn, *a, **k):
    """Run a bench, returning None on CUDA OOM instead of crashing."""
    try:
        return fn(*a, **k)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def fmt(v):
    return f"{v:8.3f} ms" if v is not None else "    OOM   "


def speed(ref_t, t):
    if ref_t is None or t is None:
        return "  n/a "
    return f"{ref_t / t:5.2f}x"


def run(cfg, device):
    B, S, d_model, d_ff, E, top_k = cfg
    print(f"\n=== B={B} S={S} d_model={d_model} d_ff={d_ff} num_expert={E} top_k={top_k} "
          f"(tokens={B*S}) ===")
    x = torch.randn(B, S, d_model, device=device)

    ref = MoE(d_model, d_ff, E, top_k).to(device)
    par = MoEParallel(d_model, d_ff, E, top_k).to(device)
    grp = MoEGrouped(d_model, d_ff, E, top_k).to(device)
    share_weights(ref, par)
    share_weights(ref, grp)

    print(f"  rel err vs loop : parallel(fp32) {fmt_err(ref, par, x)} | "
          f"grouped(bf16) {fmt_err(ref, grp, x)}")

    f = {n: safe(bench_forward, m, x, device) for n, m in [("loop", ref), ("par", par), ("grp", grp)]}
    t = {n: safe(bench_train, m, x, device) for n, m in [("loop", ref), ("par", par), ("grp", grp)]}

    print(f"  forward  : loop {fmt(f['loop'])} | parallel {fmt(f['par'])} ({speed(f['loop'], f['par'])}) "
          f"| grouped {fmt(f['grp'])} ({speed(f['loop'], f['grp'])})")
    print(f"  train    : loop {fmt(t['loop'])} | parallel {fmt(t['par'])} ({speed(t['loop'], t['par'])}) "
          f"| grouped {fmt(t['grp'])} ({speed(t['loop'], t['grp'])})")

    del ref, par, grp, x
    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    torch.manual_seed(0)

    configs = [
        # (B, S, d_model, d_ff, num_expert, top_k)
        (2,   10,  1024, 2048,  4, 2),   # the repo's example sizes
        (8,  512,  1024, 4096,  8, 2),   # realistic transformer block
        (8,  512,  1024, 4096, 16, 2),   # more experts
        (16, 1024, 1024, 4096, 32, 4),   # large batch, many experts
        (16, 1024, 1024, 4096, 64, 4),   # MoE-scale expert count
    ]
    for cfg in configs:
        run(cfg, device)
