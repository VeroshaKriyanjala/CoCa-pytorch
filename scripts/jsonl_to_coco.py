import json, argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    out = []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append({"image_id": r["image_id"], "caption": r["caption"]})
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} predictions → {args.out_json}")
if __name__ == "__main__":
    main()
