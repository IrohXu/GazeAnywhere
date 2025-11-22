import logging
from typing import Literal, Union
from functools import partial
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from detectron2.modeling import Backbone

try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_ON = True
except ImportError:
    XFORMERS_ON = False

logger = logging.getLogger(__name__)


class MetaCLIP2Tokenizer():
    def __init__(
        self, 
        tokenizer_name,
        device: Union[torch.device, str] = "cuda",
        dtype: Union[torch.dtype, str] = "float32",
    ):
        super().__init__()
        
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def tokenize(self, texts):
        return self.tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")['input_ids']


class MetaCLIP2(Backbone):
    def __init__(
        self, 
        vision_tower,
        mm_vision_select_layer=-2,
        device: Union[torch.device, str] = "cuda",
        dtype: Union[torch.dtype, str] = "float32",
    ):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.select_layer = mm_vision_select_layer
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.load_model()

    def load_model(self, device_map=None):
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, device_map=device_map).vision_model
        self.text_tower = AutoModel.from_pretrained(self.vision_tower_name, device_map=device_map).text_model
        # self.vision_tower.requires_grad_(False)
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features

    def forward(self, images, texts, masks=None, guidance=None):

        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        text_forward_outs = self.text_tower(texts.to(device=self.device))
        text_features = text_forward_outs.last_hidden_state
        
        outputs = {}
        B, HW, _ = image_features.shape

        outputs["img_feat"] = image_features[:, 1:]
        outputs["text_feat"] = text_features
        outputs["text_emb"] = text_forward_outs.pooler_output
        outputs["img_emb"] = image_forward_outs.pooler_output
        
        return outputs

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    @property
    def embed_dim(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def build_backbone_metaclip2(name, mm_vision_select_layer, **kwargs):
    return MetaCLIP2(
        vision_tower=name,
        mm_vision_select_layer=mm_vision_select_layer,
        **kwargs
    )

def build_tokenizer_metaclip2(name, **kwargs):
    tokenizer = MetaCLIP2Tokenizer(
        tokenizer_name=name,
        **kwargs
    )
    return tokenizer


# if __name__ == "__main__":
#     model = build_backbone_metaclip2(
#         name="/projects/illinois/eng/cs/jrehg/checkpoints/MetaCLIP/metaclip-2-worldwide-huge-378",
#         mm_vision_select_layer=-2,
#         dtype="float32"
#     )
#     tokenizer = build_tokenizer_metaclip2(
#         name="/projects/illinois/eng/cs/jrehg/checkpoints/MetaCLIP/metaclip-2-worldwide-huge-378",
#     )
#     model = model.cuda()
#     dummy_input = torch.randn(2, 3, 378, 378).cuda()
#     dummy_text = ["A photo of a cat. And a big apple.", "A photo of a dog."]
#     dummy_text = tokenizer.tokenize(dummy_text)
#     out = model(dummy_input, dummy_text)