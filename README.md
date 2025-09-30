# DP-AG: Action-Guided Diffusion Policy

This repository contains the official implementation of our paper:

**Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies**
*Jing Wang, Weiting Peng, Jing Tang, Zeyu Gong, Xihua Wang, Bo Tao, Li Cheng*
*[NeurIPS 2025]* 

[📄 Paper](https://jingwang18.github.io/dp-ag.github.io/) | [🌐 Project Page](https://jingwang18.github.io/dp-ag.github.io/)

---

## 📖 Overview

Existing imitation learning methods freeze perception during action sequence generation, ignoring how humans naturally refine perception through ongoing actions. **DP-AG (Action-Guided Diffusion Policy)** closes this gap by evolving observation features dynamically with action feedback.

* Latent observations are modeled via **variational inference**.
* An **action-guided SDE** evolves features, driven by the **Vector–Jacobian Product (VJP)** of diffusion noise predictions.
* A **cycle-consistent contrastive loss** aligns evolving and static latents, ensuring smooth perception–action interplay.

DP-AG significantly outperforms state-of-the-art methods on **Robomimic, Franka Kitchen, Push-T, Dynamic Push-T**, and **real-world UR5 tasks**, delivering higher success rates, faster convergence, and smoother actions.

<p align="center">
  <img src="preview.png" alt="DP-AG Overview" width="80%">
</p>

*Figure: DP-AG extends Diffusion Policy by evolving observation features through an action-guided SDE and aligning perception–action interplay with a cycle-consistent contrastive loss.*

---

## 🛠️ Installation

We follow the same setup as [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/). To reproduce simulation benchmarks, install the conda environment on a Linux machine with an NVIDIA GPU.

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge):

```bash
mamba env create -f conda_environment.yaml
```

For conda:

```bash
conda env create -f conda_environment.yaml
```

> Note: `conda_environment_macos.yaml` is only for development on macOS and does not support full benchmarks.

---

## 📥 Download Training Data

Create the `data` directory under the repo root:

```bash
mkdir data && cd data
```

Download training datasets:

```bash
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip && cd ..
```

Grab experiment configs:

```bash
wget -O image_pusht_diffusion_policy_cnn.yaml \
https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml
```


## ⚡ Quick Start with Jupyter Notebooks

We provide two Jupyter notebooks that contain the **core implementation** of DP-AG and are designed to be **easy to use and understand**:

* `PushT-Vision-Image-Action-Guided.ipynb` – Demonstrates our method on the **Push-T benchmark**.
* `Dynamic-PushT-Environment.ipynb` – Showcases our **Dynamic Push-T** environment with action–perception interplay.

👉 **We strongly suggest starting from these notebooks**, as they provide the clearest entry point for understanding and experimenting with DP-AG.

---

## 🚀 Running Experiments

Activate the conda environment and log into [wandb](https://wandb.ai):

```bash
conda activate robodiff
wandb login
```

Train with a single seed:

```bash
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
training.seed=42 training.device=cuda:0
```

Train with multiple seeds using Ray:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3
python ray_train_multirun.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
--seeds=42,43,44 --monitor_key=test/mean_score
```

---

## 📊 Evaluate Pre-trained Checkpoints

Download a checkpoint (example):

```bash
wget https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt -O data/checkpoint.ckpt
```

Run evaluation:

```bash
python eval.py --checkpoint data/checkpoint.ckpt --output_dir data/pusht_eval_output --device cuda:0
```

---

## 🤖 Real Robot Experiments

Our framework has been validated on a **UR5 robot** with **RealSense cameras** and **SpaceMouse teleoperation**.
Please refer to `demo_real_robot.py` and `eval_real_robot.py` for data collection, training, and evaluation following the same structure as Diffusion Policy.

---

## 🔧 Codebase Structure

The codebase follows [DP’s](https://diffusion-policy.cs.columbia.edu/) modular design:

* **Tasks**: dataset wrappers, environments, configs.
* **Policies**: inference + training.
* **Workspaces**: manage experiment lifecycle.

See [`train.py`](./train.py) as the entry point.

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@inproceedings{wang2025dpag,
  title={Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies},
  author={Wang, Jing and Peng, Weiting and Tang, Jing and Gong, Zeyu and Wang, Xihua and Tao, Bo and Cheng, Li},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS 2025)},
  year={2025}
}
```

---

## 🙏 Acknowledgements  

We build upon the foundational work of **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (Chi et al.).  
Their open-source code, benchmarks, and datasets enabled our development of DP-AG.  
We especially thank the authors for releasing simulation environments, vision and state-based notebooks, and experiment data.  

🔗 [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)
