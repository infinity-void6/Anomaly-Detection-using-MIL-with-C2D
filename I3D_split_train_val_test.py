import os
import random
import shutil
import torch
from torch.utils.data import Dataset, DataLoader

# Define paths
normal_dir = r"E:\MIL\Code\I3D\normal_features"
anomaly_dir = r"E:\MIL\Code\I3D\anomaly_features"

# Create output directories
output_dirs = {
    "train": {"normal": "split_I3D/train/normal", "anomaly": "split_I3D/train/anomalous"},
    "val": {"normal": "split_I3D/val/normal", "anomaly": "split_I3D/val/anomalous"},
    "test": {"normal": "split_I3D/test/normal", "anomaly": "split_I3D/test/anomalous"},
}

# Create the directories
for split, dirs in output_dirs.items():
    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)

# Load all files
normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".pt")]
anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir) if f.endswith(".pt")]

# Shuffle the files
random.seed(42)
random.shuffle(normal_files)
random.shuffle(anomaly_files)

# Compute splits
split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

def split_files(files, ratios):
    n_train = int(len(files) * ratios["train"])
    n_val = int(len(files) * ratios["val"])
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test

normal_train, normal_val, normal_test = split_files(normal_files, split_ratios)
anomaly_train, anomaly_val, anomaly_test = split_files(anomaly_files, split_ratios)

# Copy files to respective directories
for split, files in [("train", normal_train), ("val", normal_val), ("test", normal_test)]:
    for file in files:
        shutil.copy(file, output_dirs[split]["normal"])

for split, files in [("train", anomaly_train), ("val", anomaly_val), ("test", anomaly_test)]:
    for file in files:
        shutil.copy(file, output_dirs[split]["anomaly"])

print("Dataset split completed.")
print(f"Train: Normal = {len(normal_train)}, Anomaly = {len(anomaly_train)}")
print(f"Validation: Normal = {len(normal_val)}, Anomaly = {len(anomaly_val)}")
print(f"Test: Normal = {len(normal_test)}, Anomaly = {len(anomaly_test)}")

def pad_to_max_segments(feature_tensor, max_segments):
    """
    Pads the feature tensor to the max_segments along the batch/segment dimension.
    
    Args:
        feature_tensor (torch.Tensor): Input tensor of shape [B, ...].
        max_segments (int): Maximum segments to pad to.
        
    Returns:
        torch.Tensor: Padded tensor of shape [max_segments, ...].
    """
    current_segments = feature_tensor.size(0)
    if current_segments < max_segments:
        padding = torch.zeros((max_segments - current_segments, *feature_tensor.shape[1:]), device=feature_tensor.device)
        feature_tensor = torch.cat([feature_tensor, padding], dim=0)
    return feature_tensor

class FeatureDataset(Dataset):
    def __init__(self, normal_files, anomaly_files, max_segments):
        self.normal_files = normal_files
        self.anomaly_files = anomaly_files
        self.max_segments = max_segments  # Max segments for padding

    def __len__(self):
        return max(len(self.normal_files), len(self.anomaly_files))

    def __getitem__(self, idx):
        normal_idx = idx % len(self.normal_files)
        anomaly_idx = idx % len(self.anomaly_files)

        normal_feature = torch.load(self.normal_files[normal_idx])["features"]
        anomaly_feature = torch.load(self.anomaly_files[anomaly_idx])["features"]

        # Apply padding
        normal_feature = pad_to_max_segments(normal_feature, self.max_segments)
        anomaly_feature = pad_to_max_segments(anomaly_feature, self.max_segments)

        return normal_feature, anomaly_feature

# Create DataLoaders for training, validation, and testing
def create_dataloader(normal_dir, anomaly_dir, batch_size,max_segments):
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".pt")]
    anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir) if f.endswith(".pt")]
    dataset = FeatureDataset(normal_files, anomaly_files,max_segments)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Directories for train, val, and test splits
train_normal_dir = "split_I3D/train/normal"
train_anomalous_dir = "split_I3D/train/anomalous"
val_normal_dir = "split_I3D/val/normal"
val_anomalous_dir = "split_I3D/val/anomalous"
test_normal_dir = "split_I3D/test/normal"
test_anomalous_dir = "split_I3D/test/anomalous"

# Max segments for each split
max_segments_train = 509 # Max for training
max_segments_val = 253    # Max for validation
max_segments_test = 281   # Max for testing

# Batch size
batch_size = 4

# Create DataLoaders
train_loader = create_dataloader(train_normal_dir, train_anomalous_dir, batch_size, max_segments_train)
val_loader = create_dataloader(val_normal_dir, val_anomalous_dir, batch_size, max_segments_val)
test_loader = create_dataloader(test_normal_dir, test_anomalous_dir, batch_size, max_segments_test)

# Print DataLoader sizes
print(f"Train Loader: {len(train_loader)} batches")
print(f"Validation Loader: {len(val_loader)} batches")
print(f"Test Loader: {len(test_loader)} batches")