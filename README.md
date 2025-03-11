# 3D Scene Reconstruction & Semantic Understanding

This repository contains the implementation of a novel approach for **3D scene reconstruction** and **semantic scene understanding** using an **exploratory robotic system**. The goal is to enable a robot to generate an initial low-quality reconstruction of an environment, construct a scene graph, and iteratively refine it by interacting with objects (e.g., opening drawers, doors) to improve semantic understanding.

## 📌 Project Overview

This project builds upon recent advancements in **3D vision and scene representation**, primarily leveraging the concepts introduced in:
- **Spot-Light** ([Paper](https://arxiv.org/abs/2409.11870), [GitHub](http://timengelbracht.github.io/SpotLight/))
- **Spot-Compose** ([Paper](https://arxiv.org/abs/2404.12440), [GitHub](https://spot-compose.github.io/))

## 📂 Repository Structure
```
📦 project_root
├── 📂 data                 # Dataset storage
├── 📂 models               # 3D reconstruction and scene understanding models
├── 📂 scripts              # Training, evaluation, and inference scripts
├── 📂 experiments          # Results and logs from different training runs
├── 📂 utils                # Utility functions for data preprocessing, visualization, etc.
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
└── main.py                # Entry point for running experiments
---
🌟 *If you find this project useful, consider giving it a star!* 🌟
