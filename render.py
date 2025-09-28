# candidate_generation.py

import torch
import torch.nn.functional as F
from typing import List, Tuple

@torch.no_grad()
def _next_logits(coca, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Run CoCa and return logits for the next token position.
    tokens: (1, cur_len) LongTensor
    returns: (1, vocab_size)
    """
    # coca(text=tokens, images=images) -> (1, cur_len, vocab_size)
    logits = coca(text=tokens, images=images)
    return logits[:, -1, :]  # next-token logits at last position


def _apply_repetition_penalty(logits: torch.Tensor, tokens: torch.Tensor, penalty: float = 1.1):
    """
    Simple repetition penalty: downweight tokens that already appeared.
    """
    if penalty <= 1.0:
        return
    unique_tokens = tokens.unique()
    logits[..., unique_tokens] /= penalty


def _top_p_sample(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> int:
    """
    Nucleus (top-p) sampling on a single row of logits.
    """
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # keep minimum set whose cumulative prob <= top_p
    cutoff = (cumulative > top_p).float().cumsum(dim=-1) >= 1
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    choice = torch.multinomial(sorted_probs, num_samples=1)  # (1, 1)
    next_id = sorted_idx.gather(-1, choice).item()
    return next_id


@torch.no_grad()
def generate_nucleus(
    coca,
    images: torch.Tensor,
    tokenizer,
    N: int = 10,
    max_new_tokens: int = 32,
    top_p: float = 0.9,
    temperature: float = 1.0,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """
    Return N candidates via top-p sampling.
    images: (1, 3, H, W) on same device as model
    """
    device = images.device
    bos, eos = tokenizer.bos_id, tokenizer.eos_id

    candidates = []
    for _ in range(N):
        tokens = torch.tensor([[bos]], device=device, dtype=torch.long)  # (1,1)

        for _step in range(max_new_tokens):
            logits = _next_logits(coca, images, tokens)  # (1, V)
            _apply_repetition_penalty(logits, tokens[0], penalty=repetition_penalty)

            next_id = _top_p_sample(logits[0], top_p=top_p, temperature=temperature)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)

            if next_id == eos:
                break

        # decode excluding BOS and (optional) EOS
        seq = tokens[0].tolist()
        if seq and seq[0] == bos:
            seq = seq[1:]
        if seq and seq[-1] == eos:
            seq = seq[:-1]
        candidates.append(tokenizer.decode(seq))

    return candidates


@torch.no_grad()
def generate_beam(
    coca,
    images: torch.Tensor,
    tokenizer,
    beam_size: int = 5,
    max_new_tokens: int = 32,
    length_penalty: float = 0.7,
    return_top_k: int = None,  # defaults to beam_size
) -> List[Tuple[str, float]]:
    """
    Standard beam search (log-prob scoring with length penalty).
    Returns top-k (caption, score) sorted by score desc.
    """
    device = images.device
    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    V = coca.num_tokens if hasattr(coca, "num_tokens") else None  # optional

    # beams: list of (tokens_tensor, logprob_sum, is_finished)
    init = torch.tensor([[bos]], device=device, dtype=torch.long)
    beams = [(init, 0.0, False)]

    for _step in range(max_new_tokens):
        new_beams = []

        # if all finished, early stop
        if all(fin for _, _, fin in beams):
            break

        for seq, logp, finished in beams:
            if finished:
                new_beams.append((seq, logp, True))
                continue

            logits = _next_logits(coca, images, seq)  # (1, V)
            probs = F.log_softmax(logits, dim=-1)[0]  # (V,)

            # take top beam_size expansions
            topk_logp, topk_ids = torch.topk(probs, k=beam_size, dim=-1)

            for lp, tid in zip(topk_logp.tolist(), topk_ids.tolist()):
                new_seq = torch.cat([seq, torch.tensor([[tid]], device=device)], dim=1)
                new_finished = (tid == eos)
                new_beams.append((new_seq, logp + lp, new_finished))

        # keep the best beam_size by normalized score
        def norm_score(entry):
            seq, lp, fin = entry
            # +1 to avoid zero length; apply length penalty like GNMT
            L = seq.shape[1]
            return lp / ((5 + L) ** length_penalty / (5 + 1) ** length_penalty)

        new_beams.sort(key=norm_score, reverse=True)
        beams = new_beams[:beam_size]

    # finalize and decode
    out = []
    K = return_top_k or beam_size
    for seq, lp, fin in sorted(beams, key=lambda b: b[1], reverse=True)[:K]:
        ids = seq[0].tolist()
        if ids and ids[0] == bos:
            ids = ids[1:]
        if ids and ids[-1] == eos:
            ids = ids[:-1]
        out.append((tokenizer.decode(ids), lp))
    return out


# -------------------------------
# Example usage (with your CoCa setup)
# -------------------------------
if __name__ == "__main__":
    import torch
    from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
    from vit_pytorch.extractor import Extractor
    from coca_pytorch.coca_pytorch import CoCa

    # --- Build image encoder & CoCa exactly like the README ---
    vit = SimpleViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        patch_dropout = 0.5
    )
    vit = Extractor(vit, return_embeddings_only=True, detach=False)

    coca = CoCa(
        dim = 512,
        img_encoder = vit,
        image_dim = 1024,
        num_tokens = 20000,
        unimodal_depth = 6,
        multimodal_depth = 6,
        dim_head = 64,
        heads = 8,
        caption_loss_weight = 1.,
        contrastive_loss_weight = 1.,
    ).cuda().eval()

    # ---- You must provide a tokenizer compatible with your training ----
    class DummyTokenizer:
        bos_id, eos_id, pad_id = 1, 2, 0
        def decode(self, ids): return "<DECODED:" + " ".join(map(str, ids)) + ">"
        def encode(self, text): raise NotImplementedError

    tokenizer = DummyTokenizer()

    # --- One preprocessed image (B=1) ---
    images = torch.randn(1, 3, 256, 256, device="cuda")

    # (A) Nucleus sampling candidates
    samples = generate_nucleus(
        coca, images, tokenizer,
        N=10, max_new_tokens=32, top_p=0.9, temperature=0.9, repetition_penalty=1.1
    )
    print("Top-p candidates:")
    for s in samples: print(" •", s)

    # (B) Beam search candidates
    beams = generate_beam(
        coca, images, tokenizer,
        beam_size=5, max_new_tokens=32, length_penalty=0.7, return_top_k=5
    )
    print("\nBeam candidates (text, raw log-prob):")
    for txt, score in beams: print(f" • {txt}  |  {score:.2f}")
