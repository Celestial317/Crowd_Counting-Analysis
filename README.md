# Crowd Counting and Behavioral Analysis

This repository contains multiple deep learning models designed for crowd counting and behavior classification. The goal is to explore and compare different techniques such as density maps, optical flow, and motion features to handle dense crowd scenes more effectively.

## Features

- CSRNet with ConvNeXt-Tiny backbone for enhanced density map prediction
- Focus Transition Module (FTM) for adaptive channel + spatial feature integration
- Dynamic & Dilated convolutions for crowd scale adaptability
- Two-step ML pipeline for emotion and behavior classification from crowd motion
- Optical flow-based motion feature extraction
- Human segmentation + density fusion for robust region detection

## Model Performance (MAE / MSE)

| Model                          | MAE   | MSE      |
|-------------------------------|-------|----------|
| CNN Regression                | 167.03| 60788.04 |
| CSRNet (VGG-16)               | 19.52 | 1105.16  |
| CSRNet + ConvNeXt-Tiny       | **15.78** | 756.35   |
| CSRNet + Multi-Column        | 107.47| 17369.12 |
| FFNet + FTM (Dilated)        | 27.18 | 2185.14  |
| FFNet + FTM (Dynamic)        | **30.63** | 2303.36  |


## Setup

```bash
git clone https://github.com/Celestial317/Crowd_Counting-Analysis.git
cd Crowd_Counting-Analysis

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
````

## How It Works

1. Generate density maps from crowd images
2. Segment humans using DepthPro and fuse with density map
3. Use optical flow to compute motion vectors in masked regions
4. Extract motion features like speed, vorticity, confusion index, etc.
5. Predict crowd emotion → predict crowd behavior

## Credits


Authors: [Himani](https://github.com/himani2506) | [Akansh](https://github.com/DeveloperAkansh26) | [Soumya](https://github.com/Celestial317)
