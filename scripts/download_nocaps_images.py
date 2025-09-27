import json, argparse, subprocess
from pathlib import Path

def collect_ids(ann_paths):
    ids = set()
    for p in ann_paths:
        js = json.load(open(p, "r", encoding="utf-8"))
        # nocaps JSON usually has an "images" list; each item has an "id" or "open_images_id"
        for im in js.get("images", []):
            # Try common fields; adjust if your JSON uses a different key
            oid = im.get("open_images_id") or im.get("id") or im.get("image_id") or im.get("file_name")
            if oid:
                # strip extension if present
                oid = Path(str(oid)).stem
                ids.add(oid)
    return sorted(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", nargs="+", required=True, help="Paths to nocaps annotation JSON(s)")
    ap.add_argument("--out_dir", default="data/nocaps/images")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = collect_ids(args.ann)
    print(f"Total unique image IDs: {len(ids)}")

    # Use the openimages CLI to download by exact IDs (it pulls from Google’s public bucket).
    # It auto-creates subfolders; we point -d to out_dir.
    # Note: the CLI accepts many IDs; we chunk to avoid shell limits.
    chunk = 200
    for i in range(0, len(ids), chunk):
        part = ids[i:i+chunk]
        cmd = ["openimages", "download", "--image_ids"] + part + ["-d", str(out_dir)]
        print("Running:", " ".join(cmd[:5]), f"... (+{len(part)} ids)")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
