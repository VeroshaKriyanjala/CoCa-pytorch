import json, argparse, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

from rerank.coca_wrapper import CoCaWrapper
from rerank.clip_score import clip_image_embed, clip_score
from rerank.utils import length_normalize, zscore

def rerank_one(pil_img, coca, K=10, beam=5, top_p=None, temperature=1.0, alpha=0.8, len_pen=0.7):
    img_feat = clip_image_embed(pil_img)
    img_tensor = coca.preprocess_image(pil_img)
    cands = coca.generate(img_tensor, num_candidates=K, beam_size=beam, top_p=top_p, temperature=temperature)

    caps, lls = [], []
    for d in cands:
        caps.append(coca.detokenize(d["tokens"]))
        L = int(len(d["tokens"]))
        lls.append(length_normalize(float(d["logprob"]), L, len_pen))
    clips = clip_score(img_feat, caps).detach().cpu().numpy()

    lls_z, clips_z = zscore(lls), zscore(clips)
    hybrid = lls_z + alpha * clips_z
    i = int(hybrid.argmax())
    return caps[i], {"captions": caps, "ll": list(lls), "clip": list(clips), "hybrid": list(hybrid), "chosen_idx": i}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--list_file", required=True)
    ap.add_argument("--out_jsonl", default="outputs/preds.jsonl")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    Path("outputs").mkdir(parents=True, exist_ok=True)
    coca = CoCaWrapper()

    rows = []
    with open(args.out_jsonl, "w", encoding="utf-8") as fout, open(args.list_file, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Reranking"):
            fname = line.strip()
            if not fname: continue
            pil = Image.open(Path(args.images_dir) / fname).convert("RGB")
            cap, dbg = rerank_one(pil, coca, K=args.K, beam=args.beam, top_p=args.top_p,
                                  temperature=args.temperature, alpha=args.alpha)
            fout.write(json.dumps({"image_id": fname, "caption": cap, "dbg": dbg}, ensure_ascii=False) + "\n")
            rows.append([fname, cap[:80]])
    print(tabulate(rows[:10], headers=["image_id", "caption"]))
if __name__ == "__main__":
    main()
