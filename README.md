<div align="center">

# ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention

**[Xinyan Wang](https://xinyan-wang-stat.github.io/)<sup>1</sup>, [Xiaogeng Liu](https://xiaogeng-liu.com/)<sup>2</sup>, [Chaowei Xiao](https://xiaocw11.github.io/#about)<sup>2</sup>**

<sup>1</sup>University of Wisconsin-Madison &nbsp; <sup>2</sup>Johns Hopkins University

[![arXiv](https://img.shields.io/badge/arXiv-2603.22016-b31b1b.svg)](https://arxiv.org/abs/2603.22016)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://xinyan-wang-stat.github.io/ROM-LRM/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/xinyan-wang/ROM)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Abstract

Large Reasoning Models (LRMs) often reach a correct solution before their long Chain-of-Thought trace ends, yet continue with redundant verification, repeated attempts, or unnecessary exploration that wastes computation and can even overturn the correct answer. We frame this behavior as a latent productive-to-redundant transition and show that it is directly reflected in hidden states: around first-correct-solution (FCS) boundaries, late-layer representations separate efficient from overthinking tokens, while boundary-permutation and position-control baselines collapse. Based on this signal, we propose **ROM**, a model-agnostic streaming intervention framework that monitors frozen LRMs with a lightweight hidden-state detector and intervenes at well-formed reasoning boundaries. **Counterfactual Self-Correction (CSC)** augments supervision with balanced wrong→correct trajectories, preserving useful pre-FCS correction while labeling only post-FCS continuation as redundant. Across MATH500, GSM8K, AIME25, and MMLU-Pro, ROM improves the overall tradeoff on both Qwen3-8B and DeepSeek-R1-Distill-Qwen-32B (DS-32B): on Qwen3-8B, it raises accuracy from 74.47% to 74.78% and reduces response length from 4262 to 3107 tokens; on DS-32B, it raises accuracy from 68.60% to 68.72% and reduces response length from 3062 to 2319 tokens. The same FCS-derived supervision transfers across scale and training origin, suggesting a shared long-CoT boundary rather than a backbone-specific artifact. ROM is compatible with L1, removing another 20.9–21.6% tokens at zero accuracy loss. ROM also generalizes to open-ended MMLU-Pro (+1.56 pp, 35.4% shorter) and reduces wall-clock latency by 46.5%.

<p align="center">
  <img src="assets/framework.png" width="100%">
</p>

## Project Structure

```
ROM/
├── rom/                        # Core package
│   ├── models.py               # StreamingHead, Qwen3WithHead
│   ├── dataset.py              # Dataset loading & embedding cache
│   ├── train.py                # Training pipeline
│   ├── eval.py                 # Offline evaluation (vLLM)
│   ├── env.py                  # Environment setup
│   └── utils/
│       ├── math.py             # Answer extraction & correctness checking
│       └── eval_helpers.py     # Metrics, probability computation
├── configs/
│   ├── train.yaml              # Training defaults
│   └── eval.yaml               # Evaluation defaults
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11+, PyTorch >= 2.9.0, and a CUDA-capable GPU.

### Data

Training data is hosted on HuggingFace: [xinyan-wang/ROM](https://huggingface.co/datasets/xinyan-wang/ROM).

Download and place under `data/`:
```bash
# Using huggingface-cli
huggingface-cli download xinyan-wang/ROM --repo-type dataset --local-dir data
```

### Training

All parameters are in `configs/train.yaml`. Run with defaults:

```bash
python -m rom.train
```

Override via CLI:

```bash
python -m rom.train --lr 1e-4 --num_train_epochs 30
```

W&B logging is enabled by default. Disable with `--no_wandb`.

### Evaluation

We evaluate on **MATH500**, **GSM8K**, **AIME25**, and **MMLU-Pro**, served via vLLM 0.11 on a single A100 (80 GB) at temperature 0.6, top-p 0.95, top-k 20, n=3, seed 46.

```bash
python -m rom.eval
```

Override as needed:

```bash
python -m rom.eval --ckpt_path checkpoints/my_model.pt --test_data data/test_data/math500.jsonl
```

## Citation

If you find ROM useful, please cite our paper 📝 and give us a ⭐!

```bibtex
@article{wang2025rom,
  title={ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention},
  author={Wang, Xinyan and Liu, Xiaogeng and Xiao, Chaowei},
  journal={arXiv preprint arXiv:2603.22016},
  year={2025}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
