from typing import List, Dict, Any
import torch
from PIL import Image

class CoCaWrapper:
    """
    Fill the 3 TODOs to use your CoCa model directly.
    """
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TODO 1: load your CoCa + tokenizer + preprocess
        # Example (PSEUDOCODE):
        # from coca_pytorch.coca_pytorch import CoCa, get_tokenizer, get_preprocess
        # self.model = CoCa.from_pretrained("...").to(self.device).eval()
        # self.tokenizer = get_tokenizer()
        # self.preprocess = get_preprocess()

    def preprocess_image(self, pil_img: Image.Image) -> torch.Tensor:
        # TODO 2: return (1, C, H, W) tensor on self.device
        # return self.preprocess(pil_img).unsqueeze(0).to(self.device)
        raise NotImplementedError("Implement preprocess_image()")

    @torch.no_grad()
    def generate(self, image_tensor: torch.Tensor, num_candidates=10, beam_size=5,
                 top_p=None, temperature=1.0) -> List[Dict[str, Any]]:
        """
        Return a list of dicts: { "tokens": LongTensor[L], "logprob": float }
        logprob should be the sum of token log-probs (include EOS).
        """
        # TODO 3: call your model’s generate API and compute each sequence logprob.
        # You may already get logprobs from your generator; if not, request per-token scores.
        raise NotImplementedError("Implement generate()")

    def detokenize(self, token_ids) -> str:
        # TODO 4: ids -> string
        # return self.tokenizer.decode(token_ids)
        raise NotImplementedError("Implement detokenize()")
