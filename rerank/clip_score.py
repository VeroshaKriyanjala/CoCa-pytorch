import torch
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _pre, _tok = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
_model = _model.to(device).eval()

@torch.no_grad()
def clip_image_embed(pil_img):
    t = _pre(pil_img).unsqueeze(0).to(device)
    f = _model.encode_image(t)
    return (f / f.norm(dim=-1, keepdim=True)).squeeze(0)

@torch.no_grad()
def clip_text_embed(texts):
    tokens = _tok(texts).to(device)
    f = _model.encode_text(tokens)
    return f / f.norm(dim=-1, keepdim=True)

@torch.no_grad()
def clip_score(img_feat, captions):
    txt_feat = clip_text_embed(captions)
    return (txt_feat @ img_feat.unsqueeze(1)).squeeze(1)  # cosine
