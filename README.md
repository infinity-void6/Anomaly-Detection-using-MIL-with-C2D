# 🛡️ Anomaly Detection in Surveillance Videos using MIL and C2D CNN

This project implements a **weakly-supervised anomaly detection** framework for long, untrimmed surveillance videos. It uses **Multiple Instance Learning (MIL)** with a 2D Convolutional Neural Network (C2D CNN) for feature extraction and a GRU-based temporal model for segment-level anomaly scoring.

---

## 🎯 Key Features

- Detects anomalies (e.g. theft, arson, violence) in surveillance videos with only **video-level labels**.
- Uses **frame-level C2D CNN features** for efficient video representation.
- Models **temporal dependencies** using a custom **Deep GRU architecture**.
- End-to-end support for **feature extraction**, **training**, **evaluation**, and **web-based demo** using **Streamlit**.

---

## 🗂️ Project Structure

| File                          | Description                                   |
|------------------------------|-----------------------------------------------|
| `C2D_extract_features.py`     | Extract features using C2D CNN                |
| `extract_segments.py`         | Segment videos into frames                    |
| `feature_extraction_dataset.py`| Dataset loader for segment batches            |
| `C2D_split_train_val_test.py` | Dataset splitting and dataloader              |
| `Sequential_Model.py`         | GRU-based MIL anomaly detection model         |
| `Training_Model.py`           | Training script                               |
| `thresholds.py`               | ROC, AUC, threshold evaluation                |
| `website.py`                  | Streamlit demo interface                      |
| `verification.py`             | Ensemble inference module                     |
| `metadata.csv`                | Filepaths and labels for dataset              |
| `requirements.txt`            | Python dependencies                           |

---

## 📦 Installation

### 1. Clone the Repository

git clone https://github.com/your-username/Anomaly-Detection-MIL-C2D.git
cd Anomaly-Detection-MIL-C2D'

### 2. Create Environment & Install Dependencies
pip install -r requirements.txt

## 🎥 Dataset Format
Prepare your dataset as follows:

metadata.csv with:

    file_path,label
    path/to/video1.mp4,0
    path/to/video2.mp4,1
    where 0 = Anomalous, 1 = Normal.

Videos should be organized in folders and listed with full paths.

## 🚀 How to Run
1. Extract Features from Videos
  python C2D_extract_features.py
2. Split Dataset & Pad Features
  python C2D_split_train_val_test.py
3. Train the GRU Model
   python Training_Model.py
4. Evaluate on Test Set
  python thresholds.py
5. Run Web Demo
  streamlit run website.py

## 🧠 Model Details
Feature Extractor: 2D CNN from torchvision.models

Temporal Model: 16-layer GRU with optional layer norm

MIL Objective: Ranking loss with smoothness + sparsity constraints

Evaluation Metrics: AUC, PR curve, ROC, segment scoring

📈 Example Results
|Dataset |Backbone	|AUC (%)|	Remarks|
|--------|----------|-------|--------|
|UCF-Crime|	C2D CNN	|~73.8	|GRU + MIL ranking loss|
