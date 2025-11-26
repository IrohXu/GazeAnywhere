from .common.dataloader import dataloader
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.train import train
from os.path import join, basename
from torch.cuda import device_count
from modeling import backbone, models, criterion
from detectron2.config import LazyCall as L

model = L(models.AnyGazeModelMapper)()
model.backbone = L(backbone.build_backbone_dinov3txt)(
    name="dinov3_large"
)
model.tokenizer = L(backbone.build_tokenizer_dinov3txt)()
model.criterion = L(criterion.AnyGazeMapperCriterion)()
model.criterion.use_focal_loss = True
model.device = "cuda"
model.freeze_backbone = True
model.inout = True
model.patch_size = 16
model.dim = 512
model.num_layers = 6
model.image_size = 512
