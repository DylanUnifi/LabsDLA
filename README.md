# Deep Learning Architectures (DLA) Labs

**Deep Learning Architectures**  
**MSc in Artificial Intelligence, University of Florence**  
**Instructor:** Prof. Andrew D. Bagdanov

## Overview
This repository contains the implementation of various Deep Learning models and methodologies as part of the Deep Learning Architectures (DLA) Labs. The project covers a wide spectrum of deep learning domains, including Computer Vision (CNNs, ResNet, Faster-RCNN), Natural Language Processing (Transformers, BERT), Deep Reinforcement Learning (REINFORCE, DQN, PPO), as well as Out-of-Distribution (OOD) detection and model calibration.

- **Lab 1: Image Classification & Object Detection:** Exploratory Data Analysis, Feature Extraction, Fine-tuning, Visual Interpretability (Grad-CAM), and Traffic Sign Detection (Faster-RCNN).
- **Lab 2: Natural Language Processing:** Sentiment Analysis using Transformer models (e.g., BERT) and the Hugging Face ecosystem.
- **Lab 3: Deep Reinforcement Learning:** Implementation and refactoring of DRL algorithms (REINFORCE, PPO, DQN, A2C) on Gymnasium environments (e.g., CartPole, LunarLander, CarRacing).
- **Lab 4: Robustness & Calibration:** Adversarial Learning, Out-of-Distribution (OOD) Detection, and Calibration of Neural Networks.

## Implementation Details
- **Architectures**: Custom implementations of Deep Neural Networks and Reinforcement Learning agents (DQN, A2C, PPO).
- **Optimization**: Standard PyTorch optimizers (Adam, AdamW) with learning rate schedulers.
- **Hardware**: Multi-GPU support and CUDA-accelerated training via PyTorch.

## Reproducibility

The environment is fully containerized using Docker to ensure exact reproducibility across different machines.

### 1. Build Environment
```bash
git clone https://github.com/DylanUnifi/LabsDLA.git
cd LabsDLA
docker compose build
```

### 2. Configure API Keys
Copy the environment template and fill in your keys:
```bash
cp .env.example .env
```
Then edit `.env` with your actual keys:
- **WANDB_API_KEY**: Get it from [wandb.ai/settings](https://wandb.ai/settings)
- **HF_TOKEN**: Get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

> **Note:** The `.env` file is git-ignored and will never be committed. All notebooks load keys automatically via `env_setup.py`.

### 3. Run the Environment
To launch the Jupyter Notebook environment using the provided Docker configuration:
```bash
docker compose up -d
```
Then, you can access the notebooks in your browser at `http://localhost:8888`.

### 4. Training & Evaluation
You can find the training scripts and notebooks inside the respective `Lab` directories. The repository also includes `.mp4` video recordings of the trained agents (e.g., CartPole, LunarLander, CarRacing) demonstrating their performance.

All outputs, models, and generated videos are saved locally during training.
