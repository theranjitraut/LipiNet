# LipiNet
A deep learning model for classifying handwritten Devanagari characters and digits using a custom CNN architecture with **residual blocks**, **channel attention**, and **linear capsule routing**.

---

## Overview

This project implements a high-accuracy model for the **Devanagari Handwritten Character Dataset (DHCD)**, which contains 46 classes (36 consonants + 10 digits). The model leverages:
- **Dual-path stem** (standard + horizontal stroke scaffold)
- **Dense residual blocks** for feature extraction
- **Squeeze-and-Excitation (SE) channel attention**
- **Linear Capsule Routing** for class-discriminative scores
- **Gated fusion** of capsule and dense head outputs
- **Cosine-annealing learning rate schedule** and **AdamW optimizer**

---

## Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
**Requirements:**
- Python 3.8+
- TensorFlow 2.x
- PyTorch
- NumPy

### 3. Download Dataset
The model expects the [DHCD dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset) to be available at:
```
https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset
```
Or, manually download and extract the dataset to `./data/DHCD/`.

---

## Project Structure
```
.
├── data/
│   └── DHCD/
│       ├── Train/
│       └── Test/
├── results/
│   └── (saved model checkpoints)
├── README.md
└── train.py
```

---

## Training

### Configuration
- `num_classes`: 46 (36 consonants + 10 digits)
- `image_size`: 32x32 (grayscale)
- `batch_size`: 64
- `epochs`: 100
- `lr`: 5e-4 (peak learning rate)
- `weight_decay`: 1e-4 (AdamW L2 regularization)
- `label_smoothing`: 0.1
- `val_split`: 0.1

### Run Training
```bash
python train.py
```
- Models are saved to `./results/` as `.keras` files.
- Training uses **cosine-annealing LR schedule** and **early stopping** (patience=15).

---

## Model Architecture

### Key Components
1. **Stem**:
   - Dual-path: Standard 3×3 conv + horizontal (1×5) conv for Devanagari stroke detection.
   - Channel attention (SE block) for feature recalibration.

2. **Encoder**:
   - 3 stages of **dense residual blocks** (64→128→256 channels).
   - Scaffold injection: Horizontal stroke features are pooled and added to each encoder stage.

3. **Decoder Head**:
   - **Linear Capsule Routing**: Projects multi-scale features into capsule space.
   - **Dense Head**: Parallel classification head.
   - **Gated Fusion**: Learns to blend capsule and dense outputs.

4. **Optimization**:
   - **AdamW** optimizer with weight decay.
   - **Cosine-annealing LR schedule** (decays to 1e-6).
   - **Label smoothing** for better calibration.

---

## Results

| Model          | Test Accuracy | Macro F1 | Parameters          | Test Loss |
|----------------|---------------|----------|---------------------|-----------|
| LipiNet        | 99.75%        | 99.75%   | (3,863,880) ~3.9M   | 0.703    |

> **Note**: Results are run with seeds `[42]` for reproducibility.

---

<!-- ## Citation

If you use this work, please cite:
```bibtex
@misc{devanagari_dhcd,
  author = {Your Name},
  title = {Devanagari Handwritten Character Recognition with Capsule Routing and Residual Blocks},
  year = {2026},
  howpublished = {\url{https://github.com/your-repo}},
}
```

---
## Contributing
Pull requests are welcome! For major changes, open an issue first.

---
## License
[MIT](https://choosealicense.com/licenses/mit/)
```

--- -->