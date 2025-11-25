# GazeAnywhere: Gaze Target Estimation Anywhere with Concepts

* The first text and visual concept-driven gaze target estimation model.
* Defines the Promptable Gaze Target Estimation (PGE) task.
* The first Gaze Target Estimation Agent - AnyGaze Agent, which connects GazeAnywhere to Gemini 2.5 and Gemini 3.      

### [UIUC Rehg Lab](https://rehg.org/) | [Google AR](https://arvr.google.com/) | [PediaMed AI](https://pediamedai.com/)

[Xu Cao](https://www.irohxucao.com/)\*†,
[Houze Yang]()\*,
[Vipin Gunda](https://www.vipingunda.com/)\*,
[Zhongyi Zhou](https://research.google/people/zhongyizhou/)\*,
[Tianyu Xu](https://research.google/people/cady-tianyu-xu/)\*,
[Adarsh Kowdle](https://research.google/people/adarshkowdle/),
[Inki Kim](https://grainger.illinois.edu/about/directory/faculty/inkikim),
[Jim Rehg](https://rehg.org/)†

\* core contributor, † project lead

![GazeAnywhere architecture](assets/model_diagram.png?raw=true)

Estimating human gaze targets from images in-the-wild is an important and formidable task. Existing approaches primarily employ brittle, multi-stage pipelines that require explicit inputs, like head bounding boxes and human pose, in order to identify the subject of gaze analysis. As a result, detection errors can cascade and lead to failure. Moreover, these prior works lack the flexibility of specifying the gaze analysis task via natural language prompting, an approach which has been shown to have significant benefits in convenience and scalability for other image analysis tasks.

To overcome these limitations, we introduce the **Promptable Gaze Target Estimation (PGE)** task, a new end-to-end, concept-driven paradigm for gaze analysis. PGE conditions gaze prediction on flexible user text or visual prompts (e.g., "the boy in the red shirt" or "person at point [0.52, 0.48]") to identify a specific subject for gaze analysis. This approach integrates subject localization with gaze estimation and eliminates the rigid dependency on intermediate analysis stages.

We also propose **GazeAnywhere**, the first foundation model designed for PGE. **GazeAnywhere** uses a multi-layer transformer-based detector to fuse features from frozen encoders and simultaneously solves subject localization, in/out-of-frame presence, and gaze target heatmap estimation. 

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

1. **Create a new Conda environment (optional, but recommended):**

  ```bash
  conda create -n anygaze python=3.12
  conda activate anygaze
  ```

   Alternatively, you can use a virtual environment:

  ```bash
  python3.12 -m venv anygaze
  source anygaze/bin/activate  # On Windows: anygaze\Scripts\activate
  ```

2. **Install PyTorch with CUDA support:**

  ```bash
  pip3 install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```

3. **Install dependencies:**

  ```bash
  pip3 install -r requirements.txt
  ```

4. **Install detectron2:**

   Follow the [detectron2 documentation](https://detectron2.readthedocs.io/en/latest/) for installation, or use:

  ```bash
  pip3 install "git+https://github.com/facebookresearch/detectron2.git@017abbfa5f2c2a2afa045200c2af9ccf2fc6227f#egg=detectron2" --no-build-isolation
  ```

## Getting Started

⚠️ Before using GazeAnywhere, please request access to the checkpoints on the GazeAnywhere Hugging Face [repo](https://huggingface.co/IrohXu/GazeAnywhere). Once accepted, you need to be authenticated to download the checkpoints. You can do this by following the [authentication steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) (e.g., run `hf auth login` after generating an access token).

### Basic Usage

```python
TODO
```

## Examples

TODO

## Acknowledgement

Our implementation is inspired by [DINOv3](https://github.com/facebookresearch/dinov3), [DINOv2 Meets Text](https://github.com/facebookresearch/dinov2), [SAM 3](https://github.com/facebookresearch/sam3), [ViTGaze](https://github.com/hustvl/ViTGaze), [sharingan](https://github.com/idiap/sharingan), [Gaze-LLE](https://github.com/fkryan/gazelle), and [TransGesture](https://github.com/IrohXu/TransGesture). Thanks for their remarkable contributions and released code! If we missed any open-source projects or related articles, we would like to add the acknowledgement of this specific work immediately.

We would like to thank the following people for their contributions prior to GazeAnywhere: Fiona Ryan, Yuehao Song, Samy Tafasca, the authors of DINOv3 and SAM 3 at Meta, and the authors of OWLv2 at Google DeepMind. Part of our idea is inspired by their papers.     


## Collaboration   

We are welcoming technical contributors joining us in this project. Independent researchers making significant contributions (exploring new applications, training/inference acceleration, validating new components, providing more training data) in GazeAnywhere will be added into the author list of GazeAnywhere 2. We will regularly review the Pull requests and contact contributors.      

🔥PediaMed AI is hiring founder student technical staff, contact Xu Cao (xucao@pediamed.ai) if you're interested. The basic requirement is having one first-author top AI or HCI conference paper (not benchmark paper) including AAAI, ACL, CHI, CVPR, ECCV, EMNLP, ICCV, ICLR, ICML, IJCAI, NAACL, NeurIPS, UbiComp / Pervasive / IMWUT, UIST.    


## Citing GazeAnywhere    

If you use GazeAnywhere or the Gaze-Co dataset in your research, please use the following BibTeX entry.

```
@misc{cao2025gazeanywhere,
      title={Gaze Target Estimation Anywhere with Concepts},
      author={Xu Cao and Houze Yang and Vipin Gunda and Zhongyi Zhou and Tianyu Xu and Adarsh Kowdle and Inki Kim and Jim Rehg},
      year={2025},
      howpublished={\url{https://github.com/rehglab/GazeAnywhere}},
      note={Version 1.0},
}
```

