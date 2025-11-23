# GazeAnywhere: Gaze Target Estimation Anywhere with Concepts    

* The first text and visual concept-driven gaze target estimation model.     
* Define the Promptable Gaze Target Estimation (PGE) task.     
* The first Gaze Target Estimation Agent - AnyGaze Agent, connect GazeAnywhere to Gemini 3.      

### [UIUC Rehg Lab](https://rehg.org/) | [Google AR](https://arvr.google.com/) | [PediaMed AI](https://pediamedai.com/)

[Xu Cao](https://www.irohxucao.com/)\*†,
[Houze Yang]()\*,
[Vipin Gunda](https://www.vipingunda.com/)\*,
[Zhongyi Zhou](https://research.google/people/zhongyizhou/),
[Tianyu Xu](https://research.google/people/cady-tianyu-xu/),
[Adarsh Kowdle](https://research.google/people/adarshkowdle/),
[Inki Kim](https://grainger.illinois.edu/about/directory/faculty/inkikim),
[Jim Rehg](https://rehg.org/)†

\* core contributor, † project lead

![GazeAnywhere architecture](assets/model_diagram.png?raw=true) Estimating human gaze targets from images in-the-wild is an important and formidable task. Existing approaches primarily employ brittle, multi-stage pipelines that require explicit inputs, like head bounding boxes and human pose, in order to identify the subject of gaze analysis. As a result, detection errors can cascade and lead to failure. Moreover, these prior works lack the flexibility of specifying the gaze analysis task via natural language prompting, an approach which has been shown to have significant benefits in convenience and scalability for other image analysis tasks. To overcome these limitations, we introduce the **Promptable Gaze Target Estimation (PGE)** task, a new end-to-end, concept-driven paradigm for gaze analysis. PGE conditions gaze prediction on flexible user text or visual prompts (e.g., "the boy in the red shirt" or "person in point [0.52, 0.48]") to identify a specific subject for gaze analysis. This approach integrates subject localization with gaze estimation, and eliminates the rigid dependency on intermediate analysis stages. We also propose **GazeAnywhere**, the first foundation model designed for PGE. **GazeAnywhere** uses a multi-layer transformer-based detector to fuse features from frozen encoders and simultaneously solves subject localization, in/out-of-frame presence, and gaze target heatmap estimation. 

## Installation

### Prerequisites

- Python 3.11 or higher
- PyTorch 2.6 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

1. **Create a new Conda environment:**

  ```bash
  conda create -n anygaze python=3.12
  conda deactivate
  conda activate anygaze
  ```

2. **Install PyTorch with CUDA support:**

  ```bash
  pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```

3. **Basic installation:**

  ```bash
  pip install -r requirements.txt
  ```

4. **Install [detectron2](https://github.com/facebookresearch/detectron2), follow its [documentation](https://detectron2.readthedocs.io/en/latest/), or**

  ```
  pip install "git+https://github.com/facebookresearch/detectron2.git@017abbfa5f2c2a2afa045200c2af9ccf2fc6227f#egg=detectron2"
  ```

## Getting Started

⚠️ Before using GazeAnywhere, please request access to the checkpoints on the GazeAnywhere Hugging Face [repo](https://huggingface.co/IrohXu/GazeAnywhere). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

### Basic Usage

```python
TODO
```

## Examples

TODO

## Acknowledgement

Our implementation is inspired by [DINOv3](https://github.com/facebookresearch/dinov3), [DINOv2 Meets Text](https://github.com/facebookresearch/dinov2), [SAM 3](https://github.com/facebookresearch/sam3), [ViTGaze](https://github.com/hustvl/ViTGaze), [sharingan](https://github.com/idiap/sharingan), [Gaze-LLE](https://github.com/fkryan/gazelle), and [TransGesture](https://github.com/IrohXu/TransGesture). Thanks for their remarkable contribution and released code! If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.     

We would like to thank the following people for their contributions prior to GazeAnywhere: Fiona Ryan, Yuehao Song, Samy Tafasca, Authors of DINOv3 and SAM 3 in Meta, Authors of OWLv2 in Google DeepMind. Part of our idea is inspired by their papers.     

