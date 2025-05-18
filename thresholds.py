# 
import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from Sequential_Model import DeepGRUModel  # Ensure this matches your model

# === SETUP ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to the three checkpoints
checkpoint_paths = [
    r"E:\MIL\Code\best_model_val_roc_auc_0.7386.pth",
    r"E:\MIL\Code\Metrics_X3D_L_GRU\best_model_val_roc_auc_0.7103.pth",
    r"E:\MIL\Code\I3D_GRU_Metrics\best_model_val_roc_auc_0.8470.pth
]

# Validation directories
val_normal_dir = r"E:\MIL\Code\split_I3D\val\normal"
val_anomalous_dir = r"E:\MIL\Code\split_I3D\val\anomalous"

# === LOAD FEATURES ===
def load_features_from_directory(directory, label_value):
    features_list = []
    labels_list = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_path.endswith(".pt"):  # Ensure only .pt files are loaded
            data = torch.load(file_path)
            features = data["features"]  # Assuming features are stored under the key "features"
            features_list.append(features)
            labels_list.append(label_value)
    return features_list, labels_list

# Load normal and anomalous features
normal_features, normal_labels = load_features_from_directory(val_normal_dir, label_value=1)
anomalous_features, anomalous_labels = load_features_from_directory(val_anomalous_dir, label_value=0)

# Combine features and labels
all_features = normal_features + anomalous_features
all_labels = normal_labels + anomalous_labels

# === FUNCTION TO CALCULATE BEST THRESHOLD ===
def calculate_best_threshold(model, all_features, all_labels, device):
    """
    Calculate the best threshold for a given model based on F1 score.

    Args:
        model (torch.nn.Module): The model to evaluate.
        all_features (list): List of feature tensors.
        all_labels (list): List of ground truth labels.
        device (torch.device): Device for computation.

    Returns:
        float: Best threshold for the model.
    """
    all_probs = []

    with torch.no_grad():
        for features in all_features:
            features = features.to(device)
            if features.dim() > 2:
                features = features.squeeze(-1).squeeze(-1).view(-1, 2048)  # Ensure correct shape
            features = features.unsqueeze(0)  # Add batch dimension: [1, seq_len, 2048]

            # Get raw outputs and probabilities
            raw_outputs = model(features).squeeze().detach().cpu().numpy()  # [seq_len]
            probs = torch.sigmoid(torch.tensor(raw_outputs)).numpy()  # Apply Sigmoid

            # Aggregate probabilities (e.g., mean or max)
            video_prob = np.mean(probs)  # Use mean probability for the video
            all_probs.append(video_prob)  # Append the aggregated probability

    # Convert to NumPy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Ensure consistent lengths
    assert len(all_labels) == len(all_probs), "Mismatch between labels and probabilities!"

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    # Calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1

# === PROCESS EACH MODEL ===
thresholds = []
for checkpoint_path in checkpoint_paths:
    # Load the model
    model = DeepGRUModel(
        input_dim=2048,
        hidden_dim=256,
        num_layers=16,  # Match the checkpoint
        dropout=0.5,
        use_layer_norm=True  # Match the checkpoint
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")

    # Calculate the best threshold
    best_threshold, best_f1 = calculate_best_threshold(model, all_features, all_labels, device)
    thresholds.append(best_threshold)
    print(f"Best Threshold for {checkpoint_path}: {best_threshold:.4f} (F1 Score: {best_f1:.4f})")

# === COMBINE THRESHOLDS ===
# Simple average of thresholds
combined_threshold = np.mean(thresholds)
print(f"\nCombined Threshold (Average): {combined_threshold:.4f}")

# Weighted average of thresholds (optional)
weights = [0.4, 0.3, 0.3]  # Example weights for the models
combined_threshold_weighted = np.average(thresholds, weights=weights)
print(f"Combined Threshold (Weighted Average): {combined_threshold_weighted:.4f}")