from .common.dataloader import dataloader
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.train import train
from os.path import join, basename
from torch.cuda import device_count
from modeling import backbone, models, criterion
from detectron2.config import LazyCall as L

num_gpu = device_count()
ins_per_iter = 512
len_dataset = 130783
num_epoch = 10

model = L(models.GazeAnywhereModelMapper)()
model.backbone = L(backbone.build_backbone_dinov3txt)(
    name="dinov3_large"
)
model.tokenizer = L(backbone.build_tokenizer_dinov3txt)()
model.criterion = L(criterion.GazeAnywhereMapperCriterion)()
# model
# model.backbone.layerscale_init = 1
# model.backbone.mask_k_bias = True
# model.backbone.n_storage_tokens = 4
model.criterion.use_focal_loss = True
model.device = "cuda"
model.freeze_backbone = True
model.inout = True
model.patch_size = 16
model.dim = 512
model.num_layers = 6
# dataloader
dataloader = dataloader.gazeanywhere_dataset
dataloader.train.train_root = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets"
dataloader.val.val_root = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets"
dataloader.train.train_anno = "/projects/illinois/eng/cs/jrehg/users/xucao2/GazeAnywhere/datasets/anygaze_merged_single_view.json"
dataloader.val.val_anno = "/projects/illinois/eng/cs/jrehg/users/xucao2/GazeAnywhere/datasets/anygaze_merged_single_view.json"
dataloader.train.batch_size = ins_per_iter // num_gpu
dataloader.train.num_workers = dataloader.val.num_workers = 14
dataloader.train.distributed = num_gpu > 1
model.image_size = dataloader.train.input_size = dataloader.val.input_size = 512
dataloader.train.mask_scene = True
dataloader.train.mask_prob = 0.5
dataloader.train.mask_size = dataloader.train.input_size // model.patch_size
dataloader.train.max_scene_patches_ratio = 0.5
dataloader.val.batch_size = 32
dataloader.val.distributed = False
# train
train.init_checkpoint = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/gazeanywhere_dinov3txt_large_text_concept/model_final.pth"
train.output_dir = join("./output", basename(__file__).split(".")[0])
train.max_iter = len_dataset * num_epoch // ins_per_iter
train.log_period = len_dataset // (ins_per_iter * 10)
train.checkpointer.max_to_keep = 3
train.checkpointer.period = len_dataset // ins_per_iter
train.seed = 0
# optimizer
optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.99)
lr_multiplier.scheduler.typ = "cosine"
lr_multiplier.scheduler.start_value = 1
lr_multiplier.scheduler.end_value = 0.1
lr_multiplier.warmup_length = 1e-2