import torch
from pytorchvideo.models.resnet import create_resnet
import torch
from torch import nn
from extract_segments import extract_segments_from_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model Architecture (ResNet-50 Backbone for I3D)
model = create_resnet(
    input_channel=3,
    model_depth=50,  # ResNet-50 backbone
    model_num_class=400,  # Kinetics-400 classes (adjust as needed)
).to(device)

# Load the checkpoint
checkpoint_path = r".\C2D"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load model weights from 'model_state'
model.load_state_dict(checkpoint["model_state"], strict=False)  # Use strict=False if keys mismatch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify block 5 to include explicit global average pooling
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = torch.nn.Sequential(*model.blocks[:-1])  # Blocks 0-4
        self.final_block = model.blocks[5].pool  # Block 5 (unchanged)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Define global average pooling layer

    def forward(self, x):
        #print(f"Input shape: {x.shape}")  # Debug input shape
        x = self.backbone(x)  # Pass through blocks 0-4
        #print(f"After backbone shape: {x.shape}")  # Debug backbone output shape
        x=self.final_block(x)
        #print(f"After final_block shape: {x.shape}")  # Debug backbone output shape
        x = self.pool(x)  # Pass through block 5 pooling
        #print(f"After pooling shape: {x.shape}")  # Debug pooling output shape

        #x = torch.mean(x, dim=[-3, -2, -1], keepdim=True)  # Apply global average pooling
        #print(f"After global average pooling shape: {x.shape}")  # Debug GAP output shape

        return x

# Initialize the feature extractor
feature_extractor = FeatureExtractor(model)

# Define feature extraction function
def extract_features(segments, device, model=feature_extractor):
    """
    Extract features from video segments using the modified model.

    Args:
        model (torch.nn.Module): Pretrained and modified model.
        segments (list): List of video segments (tensors).
        device (torch.device): Device for computation.

    Returns:
        list: List of feature tensors.
    """
    features = []
    model.eval()
    with torch.no_grad():
        for segment in segments:
            segment = segment.unsqueeze(0).to(device)  # Add batch dimension and move to device
            feature = model(segment)  # Extract features
            features.append(feature.squeeze(0).cpu())  # Remove batch dimension and move to CPU
    return features

if __name__ == "__main__":
    file = r"....\Explosion\Explosion001_x264.mp4"
    list_segment = extract_segments_from_video(file, segment_size=16, target_shape=(320, 320), frame_skip=2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features=extract_features([list_segment[0]],device)
    print(features)
    for idx, segment in enumerate(list_segment):
        print(f"Segment {idx + 1} shape: {segment.shape}")
        break
