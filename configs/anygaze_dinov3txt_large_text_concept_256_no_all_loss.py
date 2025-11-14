from .common.dataloader import dataloader
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.train import train
from os.path import join, basename
from torch.cuda import device_count
from modeling import backbone, meta_arch, criterion
from detectron2.config import LazyCall as L

num_gpu = device_count()
ins_per_iter = 128
len_dataset = 119614
num_epoch = 30

model = L(meta_arch.AnyGazeModelMapper)()
model.backbone = L(backbone.build_backbone_dinov3txt)(
    name="dinov3_large"
)
model.tokenizer = L(backbone.build_tokenizer_dinov3txt)()
model.criterion = L(criterion.AnyGazeMapperCriterion)()
# model
# model.backbone.layerscale_init = 1
# model.backbone.mask_k_bias = True
# model.backbone.n_storage_tokens = 4
model.criterion.use_focal_loss = True
model.criterion.inout_weight = 0.0
model.criterion.weight_bbox = 0.0
model.criterion.weight_giou = 0.0
model.device = "cuda"
model.freeze_backbone = True
model.inout = True
model.patch_size = 16
model.dim = 256
# dataloader
dataloader = dataloader.anygaze_dataset_text
dataloader.train.train_root = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets"
dataloader.val.val_root = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets"
dataloader.train.train_anno = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/anygaze_train_annotations_new.txt"
dataloader.val.val_anno = "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/gaze_datasets/anygaze_gazefollow_test_annotations.txt"
dataloader.train.batch_size = ins_per_iter // num_gpu
dataloader.train.num_workers = dataloader.val.num_workers = 14
dataloader.train.distributed = num_gpu > 1
# dataloader.train.rand_rotate = 0.5
# dataloader.train.rand_lsj = 0.5
model.image_size = dataloader.train.input_size = dataloader.val.input_size = 512
dataloader.train.mask_scene = True
dataloader.train.mask_prob = 0.5
dataloader.train.mask_size = dataloader.train.input_size // model.patch_size
dataloader.train.max_scene_patches_ratio = 0.5
dataloader.val.batch_size = 32
dataloader.val.distributed = False
# train
train.init_checkpoint = "pretrained/dinov3_vitl16_dinotxt.pth"
train.output_dir = join("./output", basename(__file__).split(".")[0])
train.max_iter = len_dataset * num_epoch // ins_per_iter
train.log_period = len_dataset // (ins_per_iter * 10)
train.checkpointer.max_to_keep = 10
train.checkpointer.period = len_dataset // ins_per_iter
train.seed = 0
# optimizer
optimizer.lr = 1e-3
optimizer.betas = (0.9, 0.99)
lr_multiplier.scheduler.typ = "cosine"
lr_multiplier.scheduler.start_value = 1
lr_multiplier.scheduler.end_value = 0.1
lr_multiplier.warmup_length = 1e-2