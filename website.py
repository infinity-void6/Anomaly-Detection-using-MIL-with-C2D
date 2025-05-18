import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from verification import infer_video_ensemble  # Import the inference function
from Sequential_Model import DeepGRUModel  # Import your GRU model class
from extract_segments import extract_segments_from_video  # Function to extract video segments
from I3D_extract_features import extract_features  # Function to extract features from segments

# === SETUP ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the models
@st.cache_resource
def load_models():
    # Load Model 1
    model_1 = DeepGRUModel(
        input_dim=2048,
        hidden_dim=256,
        num_layers=16,
        dropout=0.5,
        use_layer_norm=True
    ).to(device)
    checkpoint_1 = torch.load(r"E:\MIL\Code\best_model_val_roc_auc_0.7363.pth", map_location=device)
    model_1.load_state_dict(checkpoint_1["model_state"])
    model_1.eval()

    # Load Model 3
    model_3 = DeepGRUModel(
        input_dim=2048,
        hidden_dim=256,
        num_layers=16,
        dropout=0.5,
        use_layer_norm=True
    ).to(device)
    checkpoint_3 = torch.load(r"E:\MIL\Code\best_model_val_roc_auc_0.7386.pth", map_location=device)
    model_3.load_state_dict(checkpoint_3["model_state"])
    model_3.eval()

    return model_1, model_3

model_1, model_3 = load_models()

# === STREAMLIT APP ===
st.title("Anomaly Detection in Videos")
st.write("Upload a video file to predict whether it is anomalous or normal.")

# File uploader
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# Threshold for anomaly detection
best_threshold = st.slider("Set Best Threshold", min_value=0.0, max_value=1.0, value=0.656, step=0.01)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    video_path = "temp_video.mp4"

    # Perform inference
    st.write("Processing the video...")
    combined_probability, combined_label = infer_video_ensemble(video_path, model_1, model_3, device, best_threshold)

    if combined_probability is not None:
        # Display results
        st.write(f"**Combined Probability of being Anomalous:** {combined_probability:.4f}")
        st.write(f"**Best Threshold Used:** {best_threshold}")
        st.write(f"**Predicted Label:** {'Anomalous' if combined_label == 0 else 'Normal'}")

        # Extract video segments and compute anomaly scores for plotting
        segments = extract_segments_from_video(video_path, segment_size=16, target_shape=(320, 320), frame_skip=2)
        if segments:
            features = extract_features(segments, device=device)
            features = torch.stack(features).unsqueeze(0).to(device)
            with torch.no_grad():
                scores_1 = model_1(features).squeeze(0).squeeze(-1).cpu().numpy()
                scores_3 = model_3(features).squeeze(0).squeeze(-1).cpu().numpy()

            # Combine scores
            combined_scores = (scores_1 + scores_3) / 3

            # Plot the anomaly scores
            st.write("### Anomaly Scores for Each Segment")
            fig, ax = plt.subplots()
            ax.plot(range(1, len(combined_scores) + 1), combined_scores, marker="o", label="Anomaly Score")
            ax.axhline(y=best_threshold, color="r", linestyle="--", label="Threshold")
            ax.set_xlabel("Segment Number")
            ax.set_ylabel("Anomaly Score")
            ax.set_title("Anomaly Scores Across Video Segments")
            ax.legend()
            st.pyplot(fig)
    else:
        st.write("An error occurred during inference. Please try again.")

# Clean up temporary file
import os
if os.path.exists("temp_video.mp4"):
    os.remove("temp_video.mp4")