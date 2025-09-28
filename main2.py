import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

# --- Vision encoder (ViT) ---
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor

# --- CoCa ---
from coca_pytorch.coca_pytorch import CoCa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Tokenizer ---------------------------------------------------------------
# Using BERT tokenizer for convenience; it has [CLS] (start), [SEP] (end), [PAD]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
pad_id = tokenizer.pad_token_id          # usually 0 for BERT
cls_id = tokenizer.cls_token_id          # [CLS]
sep_id = tokenizer.sep_token_id          # [SEP]

assert cls_id is not None and sep_id is not None, "Tokenizer must have [CLS] and [SEP]."

# 2) Image preprocessing -----------------------------------------------------
img_path = "./Coca/images/golden.jpeg"
img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

image_tensor = transform(img).unsqueeze(0).to(DEVICE)   # (1, 3, 256, 256)

# 3) Build image encoder (ViT) ----------------------------------------------
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
vit = Extractor(vit, return_embeddings_only=True, detach=False).to(DEVICE).eval()

# 4) Build CoCa --------------------------------------------------------------
coca = CoCa(
    dim = 512,
    img_encoder = vit,
    image_dim = 1024,
    num_tokens = vocab_size,        # MUST match tokenizer vocab size
    unimodal_depth = 6,
    multimodal_depth = 6,
    dim_head = 64,
    heads = 8,
    caption_loss_weight = 1.0,
    contrastive_loss_weight = 1.0,
    pad_id = pad_id
).to(DEVICE).eval()

# 5) Greedy decoding loop ----------------------------------------------------
@torch.no_grad()
def generate_caption(model: CoCa, tokenizer: AutoTokenizer, image: torch.Tensor,
                     max_len: int = 32, prompt_ids=None):
    """
    model: CoCa, in eval()
    image: (1, 3, H, W)
    prompt_ids: optional tensor (1, t) to prime decoding; default starts with [CLS]
    """
    if prompt_ids is None:
        # start with [CLS]
        tokens = torch.tensor([[tokenizer.cls_token_id]], device=image.device, dtype=torch.long)
    else:
        tokens = prompt_ids.to(image.device)

    for _ in range(max_len):
        # logits: (1, seq_len, vocab)
        logits = model(text=tokens, images=image)  # no loss, just forward
        next_logits = logits[:, -1, :]            # last position
        next_id = next_logits.argmax(dim=-1)      # greedy

        tokens = torch.cat([tokens, next_id.unsqueeze(-1)], dim=1)

        # stop if we hit [SEP]
        if next_id.item() == tokenizer.sep_token_id:
            break

    # Drop the initial [CLS] and stop at [SEP] (if present)
    seq = tokens[0].tolist()
    if seq and seq[0] == tokenizer.cls_token_id:
        seq = seq[1:]
    if tokenizer.sep_token_id in seq:
        sep_index = seq.index(tokenizer.sep_token_id)
        seq = seq[:sep_index]

    return tokenizer.decode(seq, skip_special_tokens=True)

# 6) Run it ------------------------------------------------------------------
caption = generate_caption(coca, tokenizer, image_tensor, max_len=32)
print("Generated caption:", caption)
