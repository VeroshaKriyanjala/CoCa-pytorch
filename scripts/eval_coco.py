import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)   # nocaps annotations (COCO-style)
    ap.add_argument("--pred", required=True)  # our predictions (COCO-style)
    args = ap.parse_args()

    coco = COCO(args.ann)
    cocoRes = coco.loadRes(args.pred)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    print("== Overall ==")
    for m, v in cocoEval.eval.items():
        print(f"{m:10s}: {v:.4f}")

if __name__ == "__main__":
    main()
