# div2KVF

[![Dataset](https://img.shields.io/badge/dataset-752%20images-blue)]()
[![Size](https://img.shields.io/badge/size-~71%20GB-informational)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-3776AB?logo=python&logoColor=white)]()
[![Source](https://img.shields.io/badge/source-DIV2K-orange)](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

A multi-task image processing benchmark dataset built on top of the 752 training
images from DIV2K. Each image has been processed by **multiple state-of-the-art
models per task**, compared using a quantitative metric, and **only the best
result is kept**. All outputs are provided as 16-bit or 8-bit PNGs with full
metadata in a per-image `meta.json`.

---

## Table of contents

1. [Overview](#overview)
2. [Dataset structure](#dataset-structure)
3. [Tasks and models](#tasks-and-models)
4. [Statistics](#statistics)
5. [Download](#download)
6. [Quick start](#quick-start)
7. [Reproducing the dataset](#reproducing-the-dataset)
8. [Citation](#citation)
9. [License](#license)

---

## Overview

JPEG AI is the first international standard for learning-based image
compression. It jointly optimizes human visual quality (reconstruction) and
machine vision performance (downstream tasks).

A key limitation of current learned compression systems is that modules
(encoder, decoder, vision heads) are typically trained separately, preventing
global optimization. **div2KVF** addresses this by providing a unified
multi-task dataset where a single input image is paired with multiple
AI-generated ground truths — usable for end-to-end MTL training at no manual
annotation cost.

For every task we compared two models and recorded both scores. The retained
output is the per-image winner (or the union for foreground/segmentation).

---

## Dataset structure

```
div2KVF/
├── 0001/
│   ├── hr.png                # original DIV2K HR image
│   ├── sr_x2.png             # best super-resolution x2 (HAT or MambaIRv2)
│   ├── awb.png               # best automatic white balance (FC4 or DeepWB)
│   ├── depth.png             # best depth map, 16-bit (DAv2 or MiDaS)
│   ├── seg.png               # semantic segmentation mask
│   ├── fg.png                # foreground / salient object mask
│   ├── blur_h.png            # motion blur - horizontal
│   ├── blur_v.png            # motion blur - vertical
│   ├── blur_d.png            # motion blur - diagonal
│   ├── poisson_l1.png        # Poisson noise, peak=255
│   ├── poisson_l2.png        # Poisson noise, peak=100
│   ├── poisson_l3.png        # Poisson noise, peak=50
│   ├── poisson_l4.png        # Poisson noise, peak=20
│   ├── poisson_l5.png        # Poisson noise, peak=5
│   ├── noisy_l1.png          # Gaussian noise, level 1
│   ├── noisy_l2.png          # Gaussian noise, level 2
│   ├── noisy_l3.png          # Gaussian noise, level 3
│   ├── noisy_l4.png          # Gaussian noise, level 4
│   ├── noisy_l5.png          # Gaussian noise, level 5
│   └── meta.json             # all scores, methods, captions, tags, labels
├── 0002/
│   └── ...
├── 0752/
│   └── ...
├── dataset_info.json         # global summary, schema and statistics
└── index.csv                 # flat CSV of all image IDs
```

Each folder is self-contained: 20 files per image (19 PNGs + `meta.json`).

See [`docs/schema.md`](./docs/schema.md) for the full `meta.json` schema.

---

## Tasks and models

### Super-Resolution x2

Winner selected per image by PSNR + SSIM.

| Model        | Type                          | Weights                                | Paper                                              |
|--------------|-------------------------------|----------------------------------------|----------------------------------------------------|
| HAT          | Hybrid Attention Transformer  | `HAT_SRx2_ImageNet-pretrain.pth`       | [arXiv:2309.05239](https://arxiv.org/abs/2309.05239) |
| MambaIRv2    | State-space model (Base)      | `mambairv2_classicSR_Base_x2.pth`      | [arXiv:2412.04237](https://arxiv.org/abs/2412.04237) |

| Metric     | HAT     | MambaIRv2 | Winner |
|------------|---------|-----------|--------|
| Mean PSNR  | 26.29   | **27.16** | MambaIRv2 |
| Mean SSIM  | 0.859   | **0.874** | MambaIRv2 |
| Wins       | 24      | **728**   | MambaIRv2 (96.8%) |

### Automatic White Balance

Winner selected per image by luminance-invariant chroma error
`L2(mean_BGR / mean_lum - 1)` — penalises colour cast without rewarding
brightness boost (lower is better).

| Model    | Type                                       | Notes                                  | Paper                                              |
|----------|--------------------------------------------|----------------------------------------|----------------------------------------------------|
| FC4      | SqueezeNet + Confidence-Weighted Pooling   | Global illuminant; trained on ColorChecker. Correction: `hr / (illuminant * sqrt(3))`, normalize, gamma 1/2.2. | [arXiv:1804.06197](https://arxiv.org/abs/1804.06197) |
| DeepWB   | Spatial per-pixel estimator                | Direct image-to-image translation      | [arXiv:2010.10510](https://arxiv.org/abs/2010.10510) |

| Metric             | FC4       | DeepWB | Winner |
|--------------------|-----------|--------|--------|
| Mean chroma error  | **0.107** | 0.168  | FC4    |
| Wins               | **522**   | 230    | FC4 (69.4%) |

### Depth Estimation

Winner selected per image by Sobel edge-alignment Pearson correlation
(higher is better).

| Model               | Type                              | Weights / handle                                  | Paper                                              |
|---------------------|-----------------------------------|---------------------------------------------------|----------------------------------------------------|
| Depth-Anything V2   | ViT-Small backbone                | `depth-anything/Depth-Anything-V2-Small-hf`       | [arXiv:2406.09414](https://arxiv.org/abs/2406.09414) |
| MiDaS DPT_Large     | DPT with ViT-L (OpenAI CLIP)      | `intel-isl/MiDaS` via `torch.hub`                 | [arXiv:2103.13413](https://arxiv.org/abs/2103.13413) |

| Metric                     | DAv2      | MiDaS  | Winner |
|----------------------------|-----------|--------|--------|
| Mean edge-alignment (Pearson) | **0.221** | 0.184 | DAv2   |
| Wins                       | **669**   | 83     | DAv2 (89.0%) |

Output format: 16-bit PNG, inverse disparity encoding.

### Image Captioning

Winner selected per image by CLIP ViT-B/32 image-text cosine similarity
(higher is better).

| Model      | Type                                       | Weights / handle                  | Paper                                              |
|------------|--------------------------------------------|-----------------------------------|----------------------------------------------------|
| BLIP-2     | Vision-language with Q-Former + OPT-2.7B   | `Salesforce/blip2-opt-2.7b`       | [arXiv:2301.12597](https://arxiv.org/abs/2301.12597) |
| Tag2Text   | Tag-guided captioning, Swin-14M backbone   | `xinyu1205/tag2text` (`tag2text_swin_14m.pth`) | [arXiv:2303.05657](https://arxiv.org/abs/2303.05657) |

| Metric          | BLIP-2     | Tag2Text  | Winner   |
|-----------------|------------|-----------|----------|
| Mean CLIP score | 29.51      | **30.01** | Tag2Text |
| Wins            | 336        | **416**   | Tag2Text (55.3%) |

### Other tasks (single output)

| Task          | Strategy                                                |
|---------------|---------------------------------------------------------|
| Segmentation  | Ensemble of DeepLabV3+ and FCN (ResNet-50 backbone)     |
| Foreground    | Union of MobileNet and ResNet50 salient object detection |
| Motion blur   | 3 directions: horizontal, vertical, diagonal            |
| Poisson noise | 5 SNR levels — peak = 255 / 100 / 50 / 20 / 5           |
| Gaussian noise| 5 levels of increasing standard deviation               |

---

## Statistics

| Item                  | Value                |
|-----------------------|----------------------|
| Total images          | 752                  |
| Files per image       | 20                   |
| Total size            | ~71 GB               |
| Source                | DIV2K Train HR       |
| Anonymization applied | Yes (faces removed)  |

See `dataset_info.json` at the dataset root for the machine-readable summary,
and [`docs/anonymization.md`](./docs/anonymization.md) for the anonymization
procedure.

---

## Download

The full dataset is hosted on Google Drive as a single archive.

**[Download div2KVF.zip on Google Drive](https://drive.google.com/your-link-here)**

After downloading:

```bash
unzip div2KVF.zip
```

For maintainers: see [`docs/RELEASE_GUIDE.md`](./docs/RELEASE_GUIDE.md) for the
end-to-end packaging and upload procedure.

---

## Quick start

```python
import json
import cv2
from pathlib import Path

# Pick any image folder
root = Path("div2KVF/0046")
meta = json.loads((root / "meta.json").read_text())

# Load HR and best SR
hr = cv2.imread(str(root / "hr.png"))
sr = cv2.imread(str(root / meta["tasks"]["sr"]["file"]))

# Caption (best of BLIP-2 / Tag2Text per image)
print("Caption:", meta["tasks"]["caption"]["text"])
print("Method:",  meta["tasks"]["caption"]["method"])

# Depth: load the 16-bit PNG (inverse disparity)
depth = cv2.imread(str(root / "depth.png"), cv2.IMREAD_UNCHANGED)
print("Depth dtype:", depth.dtype, "shape:", depth.shape)

# Classification: top-1 from two backbones
cls = meta["tasks"]["classification"]
for name, pred in cls.items():
    print(f"{name:14} -> {pred['label']:30} ({pred['score']:.3f})")
```

A dataset-wide iteration helper:

```python
import json
from pathlib import Path

ROOT = Path("div2KVF")
with open(ROOT / "dataset_info.json") as f:
    info = json.load(f)

print(f"{info['name']} v{info['version']} - {info['num_images']} images")

for folder in sorted(ROOT.glob("[0-9]" * 4)):
    meta = json.loads((folder / "meta.json").read_text())
    # do something with meta...
```

---

## Reproducing the dataset

Hardware used: NVIDIA RTX 3090 (24 GB) on EURECOM's `gravette` server.

### 1. Setup

```bash
git clone https://github.com/ibraguim-jalmourzaev/div2KVF.git
cd div2KVF

conda create -n div2kvf python=3.8 -y
conda activate div2kvf

pip install -r requirements.txt
```

### 2. Download source images

```bash
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip
```

Apply the anonymization step (manual face removal) — see
[`docs/anonymization.md`](./docs/anonymization.md). 800 images become 752.

### 3. Pretrained weights

| Task    | Weights                                            | Source |
|---------|----------------------------------------------------|--------|
| HAT     | `HAT_SRx2_ImageNet-pretrain.pth`                   | XPixelGroup/HAT releases (Google Drive) |
| MambaIR | `mambairv2_classicSR_Base_x2.pth`                  | csguoh/MambaIR model zoo |
| DAv2    | Auto-downloaded via HuggingFace transformers       | `depth-anything/Depth-Anything-V2-Small-hf` |
| MiDaS   | Auto-downloaded via `torch.hub`                    | `intel-isl/MiDaS` |
| FC4     | Pretrained on ColorChecker                         | original repo |
| DeepWB  | Public weights                                     | original repo |
| BLIP-2  | Auto-downloaded                                    | `Salesforce/blip2-opt-2.7b` |
| Tag2Text| `tag2text_swin_14m.pth`                            | `xinyu1205/tag2text` |

### 4. Run inference scripts

Each task is independent and writes its outputs + updates each image's
`meta.json`. Suggested order:

```
scripts/run_sr.py                  # HAT + MambaIR + per-image winner
scripts/run_awb.py                 # FC4 + DeepWB + winner
scripts/run_depth.py               # DAv2 + MiDaS + winner
scripts/run_seg_fg.py              # segmentation + foreground
scripts/run_caption.py             # BLIP-2 + Tag2Text + CLIP scoring
scripts/run_classification.py      # ResNet50 + EfficientNet-B4
scripts/run_blur.py                # motion blur kernels
scripts/run_noise.py               # Poisson + Gaussian
scripts/build_dataset_info.py      # aggregates statistics across all images
```

> Note: this repository ships dataset, schema, and documentation. The
> inference scripts are kept private to the EURECOM working tree; ask the
> authors for access if you need to regenerate the data.

---

## Citation

```bibtex
@misc{div2kvf_2026,
  author       = {*****},
  title        = {div2KVF: A Multi-Task Image Processing Benchmark Dataset},
  year         = {2026},
  howpublished = {EURECOM Semester Project},
  url          = {https://github.com/ibraguim-jalmourzaev/div2KVF}
}
```

Built on top of:

```bibtex
@inproceedings{agustsson2017div2k,
  author    = {Agustsson, Eirikur and Timofte, Radu},
  title     = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
  booktitle = {CVPRW},
  year      = {2017}
}
```

---

## License

* **Repository content (code, documentation)**: MIT — see [`LICENSE`](./LICENSE).
* **Generated dataset (annotations, masks, depth maps, captions)**: CC-BY-4.0 —
  see [`LICENSE-DATA`](./LICENSE-DATA).
* **Source HR images**: governed by the original DIV2K terms — see
  https://data.vision.ee.ethz.ch/cvl/DIV2K/.

---

## Acknowledgments

* EURECOM Imaging Security Group — compute access and supervision.
* The authors of HAT, MambaIRv2, FC4, DeepWB, Depth Anything V2, MiDaS,
  DeepLabV3+, FCN, BLIP-2, Tag2Text, ResNet, MobileNet, EfficientNet — for
  releasing their models and code.
* DIV2K dataset — for the source HR images.
