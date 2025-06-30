from detectron2.config import LazyCall as L

from modeling import backbone, patch_attention, meta_arch, head, criterion

model = L(meta_arch.GazeModelMapper)()
model.backbone = L(backbone.build_backbone)(
    name="small", out_attn=[2, 5, 8, 11]
)
model.criterion = L(criterion.GazeModelCriterion)()
model.device = "cuda"
