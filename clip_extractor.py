import torch, cv2, numpy as np, open_clip
import torchvision.transforms as T
from PIL import Image

class CLIPFeatureExtractor:
    """
    OpenCLIP image encoder
    • default backbone  : ViT-B/32  (768-D)
    • extract(img)      : single frame  → 1×768
    • extract_batch()   : list[img]    → Nx768  (fast)
    """
    def __init__(self,
                 model_name="ViT-B-32",
                 pretrained="laion2b_s34b_b79k",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device
        )
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466,0.4578275,0.40821073),
                        std =(0.26862954,0.26130258,0.27577711))
        ])
        with torch.no_grad():
            self.dim = int(self.model.encode_image(
                torch.zeros(1,3,224,224).to(device)).shape[-1])

    # -------- single frame --------
    def extract(self, img: np.ndarray) -> np.ndarray:
        if img.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            v = self.model.encode_image(
                self.transform(pil).unsqueeze(0).to(self.device))
            v = v / v.norm(dim=-1, keepdim=True)
        return v.squeeze().cpu().numpy().astype(np.float32)

    # -------- batch of frames -----
    def extract_batch(self, imgs) -> np.ndarray:
        """imgs = list of BGR numpy arrays"""
        if not imgs:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = torch.stack([
            self.transform(Image.fromarray(
                cv2.cvtColor(cv2.resize(im,(128,128)), cv2.COLOR_BGR2RGB)))
            for im in imgs]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)
