import argparse, json, re
from pathlib import Path
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

SPECIAL_TOKENS = ("<start_of_text>", "<end_of_text>")

def strip_special_tokens(s: str) -> str:
    s = s.strip()
    for t in SPECIAL_TOKENS:
        s = s.replace(t, "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def main(ann_path: str, pred_path: str, out_json: str):
    coco = COCO(ann_path)

    preds = json.load(open(pred_path, "r"))
    for p in preds:
        p["caption"] = strip_special_tokens(p["caption"])

    # Keep predictions that match images in the GT
    valid_ids = {img["id"] for img in coco.dataset["images"]}
    preds = [p for p in preds if p["image_id"] in valid_ids]
    if not preds:
        raise SystemExit("No predictions match GT image ids.")

    tmp_path = Path(pred_path).with_suffix(".filtered.json")
    json.dump(preds, open(tmp_path, "w"))

    coco_res = coco.loadRes(str(tmp_path))
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params["image_id"] = coco_res.getImgIds()
    coco_eval.evaluate()

    # Print ONLY CIDEr (and a few extras for sanity)
    print("\n=== Overall ===")
    for k in ("CIDEr", "Bleu_4", "METEOR", "SPICE", "ROUGE_L"):
        if k in coco_eval.eval:
            val = coco_eval.eval[k]
            print(f"{k:7s}: {val:.3f}" if isinstance(val, (int, float)) else f"{k}: {val}")

    # Per-image CIDEr (first 10)
    print("\n=== Per-image (first 10) ===")
    for i, (img_id, m) in enumerate(coco_eval.imgToEval.items()):
        if "CIDEr" in m:
            print(f"{img_id}: CIDEr={m['CIDEr']:.3f}")
        if i >= 9:
            break

    # Save detailed JSON
    out = {"overall": coco_eval.eval, "per_image": coco_eval.imgToEval}
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"\nSaved -> {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to nocaps_val_4500_captions.json")
    ap.add_argument("--pred", required=True, help="Path to your preds JSON")
    ap.add_argument("--out", default="cider_results.json")
    a = ap.parse_args()
    main(a.ann, a.pred, a.out)
