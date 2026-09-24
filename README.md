# Lost in Interpolation: Spherical Soft-Masking (SLERP-SM) for Diffusion Language Models

Official PyTorch implementation of **Spherical Soft-Masking (SLERP-SM)** for masked diffusion language models (MDLMs), from the paper:

> **[Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models](https://arxiv.org/abs/2608.06529)**  
> *Conference on Language Modeling (COLM 2026)*

**Paper:** [arXiv:2608.06529](https://arxiv.org/abs/2608.06529)

<div align="center">
  <img src='./assets/slerp_vs_lerp.png' width="85%"/>
</div>

---

## Overview

Soft-masking provides predictive feedback by continuously blending the mask token embedding with a superposition of top-k predictions. However, standard linear interpolation (**LERP-SM**) suffers from a severe **norm-collapse failure**:

* **Hyperspherical Geometry**: Language model token embeddings concentrate on a spherical shell in high dimensions. Straight-line interpolation (LERP) cuts *through* the hypersphere, systematically shrinking embedding norms by 20–40% and feeding out-of-distribution inputs to the backbone.
* **Spherical Soft-Masking (SLERP-SM)**:
  1. **Fréchet (Karcher) Mean**: Aggregates top-$k$ normalized predictions on the unit sphere $\mathcal{S}^{d-1}$.
  2. **Geodesic Interpolation (SLERP)**: Interpolates along the great-circle arc between the mask embedding and the Fréchet mean.
  3. **Norm Preservation**: Rescales the interpolated direction back to the original mask embedding norm, preventing norm collapse and gradient explosion.

---

## Quickstart

This repository contains implementations for both **Language Modeling** and **Code Generation**:

### 1. Language Modeling (`language/`)
Built on MDLM / Duo:
```bash
cd language
conda create -n sm-env python=3.12 -y && conda activate sm-env
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation

# Finetune with SLERP-SM (multi-GPU)
NUM_GPUS=2 BASE_CKPT=/path/to/mdlm.ckpt bash scripts/run_slerp_multigpu.sh
```
See [`language/README.md`](./language/README.md) for full pretraining, evaluation, and Modal / Slurm scripts.

### 2. Code Generation (`coding/`)
Built on Dream-7B with PEFT / DoRA:
```bash
cd coding/train
python train.py "./configs/base_configs/config_1.json" "./configs/datasets_softmasking/config_coding.json"
```
See [`coding/README.md`](./coding/README.md) for fine-tuning and HumanEval / EvalPlus evaluation.

---

## Model Checkpoints

Pretrained checkpoints for Language Modeling (OpenWebText) are available on Hugging Face:
👉 **[HuggingFace Checkpoints (lavanyanigam/soft-masking-checkpoints)](https://huggingface.co/lavanyanigam/soft-masking-checkpoints)**


## Acknowledgements
This codebase builds upon [Duo](https://github.com/s-sahoo/duo), [ReMDM](https://github.com/kuleshov-group/remdm), [Dream-7B](https://github.com/DreamLM/Dream), and [Soft-Masked DLMs](https://openreview.net/forum?id=Gba02UMvrG).
