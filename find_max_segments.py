import torch
import os

def find_max_segments(data_dir):
    """
    Finds the maximum number of segments in the feature tensors for the given directory.

    Args:
        data_dir (str): Path to the directory containing feature files.

    Returns:
        int: Maximum number of segments.
        str: File name of the tensor with the max segments.
    """
    max_segments = 0
    max_file = None

    feature_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]

    for file in feature_files:
        features = torch.load(file)["features"]  # Load features
        num_segments = features.shape[0]  # Number of segments is the first dimension (B)
        if num_segments > max_segments:
            max_segments = num_segments
            max_file = file

    return max_segments, max_file

# Directories for splits
splits = {
    "train": {"normal": r"E:\MIL\Code\split_I3D\train\normal", "anomaly":r"E:\MIL\Code\split_I3D\train\anomalous"},
    "val": {"normal": r"E:\MIL\Code\split_I3D\val\normal", "anomaly": r"E:\MIL\Code\split_I3D\val\anomalous"},
    "test": {"normal": r"E:\MIL\Code\split_I3D\test\normal", "anomaly": r"E:\MIL\Code\split_I3D\test\anomalous"}
}

# Find maximum segments for each split
for split_name, dirs in splits.items():
    for label, dir_path in dirs.items():
        max_segments, max_file = find_max_segments(dir_path)
        print(f"Max segments in {split_name} {label}: {max_segments} (File: {max_file})")