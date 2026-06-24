from transformers import AutoTokenizer

qwen_tok = AutoTokenizer.from_pretrained("model/tokenizer/qwen3_6", use_fast=True)
gemma_tok = AutoTokenizer.from_pretrained("model/tokenizer/gemma4", use_fast=True)
lfm2_5_tok = AutoTokenizer.from_pretrained("model/tokenizer/LFM2_5", use_fast=True)
lfm2_5_vl_tok = AutoTokenizer.from_pretrained("model/tokenizer/LFM2_5_VL", use_fast=True)

samples = [
    "The eigenvalues of A determine stability.",
    "我正在训练一个小型语言模型。",
    "设 A∈ℝ^{m×n}, 求 ||Ax-b||_2^2 的最小值。",
    r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}",
    "∀ε>0, ∃δ>0, |x-a|<δ ⇒ |f(x)-f(a)|<ε",
]

for name, tok in [("qwen", qwen_tok), ("gemma4", gemma_tok), ("lfm 2.5", lfm2_5_tok), ("lfm 2.5 VL", lfm2_5_vl_tok)]:
    print("=" * 40)
    print(name, "vocab:", len(tok))

    for s in samples:
        ids = tok.encode(s, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=False)

        print(repr(s))
        print("tokens:", len(ids))
        print("chars/token:", len(s) / max(len(ids), 1))
        print("roundtrip:", decoded == s)
        print()