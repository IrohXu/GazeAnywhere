from src.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
    pretrained=True,
    weights="/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/pretrained/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    backbone_weights="/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)

import urllib
from PIL import Image

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
img_pil = load_image_from_url(EXAMPLE_IMAGE_URL)

import torch
from src.dinov3.data.transforms import make_classification_eval_transform

image_preprocess = make_classification_eval_transform()
image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).cuda()
texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
class_names = ["dog", "chair", "bowl", "tupperware"]
tokenized_texts_tensor = tokenizer.tokenize(texts).cuda()
model = model.cuda()
with torch.autocast('cuda', dtype=torch.float):
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(tokenized_texts_tensor)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (
    text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
)
print(similarity) 