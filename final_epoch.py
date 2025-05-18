"""
# epoch_training_test.py

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from split_data_train_val_test import create_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import warnings
from sklearn.metrics import precision_recall_curve,f1_score,roc_curve,auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

random.seed(42)
# Assuming SequentialMILModel is already implemented
from Sequential_Model import SequentialMILModel
from Sequential_Model import DeepGRUModel

# Device configuration
device = torch.device("cuda")

# 1. Initialize Model
model = DeepGRUModel(
            input_dim=2048,  # Feature size
            hidden_dim=256,  # GRU hidden size
            num_layers=16,   # Number of GRU layers
            dropout=0.5,     # Dropout rate
            use_layer_norm=True  # Use layer normalization
        ).to(device)

# # 1. Initialize Model
# model = SequentialMILModel(
#             input_dim=2048,  # Feature size
#             hidden_dim=512,  # GRU hidden size
#         ).to(device)

# 2. Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Scheduler: Reduces learning rate when validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# 3. Define Loss Function  # Replace with your ranking loss function

# 4. Gradient Scaler for Mixed Precision
scaler = GradScaler()

# Data Loaders
# Directories for train, val, and test splits
train_normal_dir = "split_I3D/train/normal"
train_anomalous_dir = "split_I3D/train/anomalous"
val_normal_dir = "split_I3D/val/normal"
val_anomalous_dir = "split_I3D/val/anomalous"
test_normal_dir = "split_I3D/test/normal"
test_anomalous_dir = "split_I3D/test/anomalous"

# Max segments for each split
max_segments_train = 509  # Max for training
max_segments_val = 253  # Max for validation
max_segments_test = 281   # Max for testing


# Batch size
batch_size = 4

# Create DataLoaders
train_loader = create_dataloader(train_normal_dir, train_anomalous_dir, batch_size, max_segments_train)
val_loader = create_dataloader(val_normal_dir, val_anomalous_dir, batch_size, max_segments_val)
test_loader=create_dataloader(test_normal_dir,test_anomalous_dir,batch_size, max_segments_test)


def ranking_loss(scores, labels, batch_size, lamda_sparsity=8e-5, lamda_smooth=8e-5, k_ratio=0.2):
    """
    # np.hanning loss with Binary Cross-Entropy for weakly-supervised MIL anomaly detection.
    # Handles label-output consistency with top-k selection.
"""
    num_segments = scores.shape[0] // batch_size  # Segments per video
    total_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)

    for i in range(batch_size):
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i].float()  # Single video-level label

        # Compute feature magnitudes
        feature_magnitudes = torch.norm(video_scores, p=2, dim=-1)

        # Ensure feature_magnitudes is not a scalar
        if feature_magnitudes.dim() == 0:  
            k = 1  # Default to 1 snippet
        else:
            k = max(1, int(len(feature_magnitudes) * k_ratio))  # Select top k% of snippets

        # Top-k selection
        top_k_features, _ = torch.topk(feature_magnitudes, k)

        # Compute BCE loss for the mean of top-k features
        mean_score = top_k_features.mean()
        bce_loss = F.binary_cross_entropy_with_logits(mean_score, video_label)

        # Sparsity and Smoothness
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum(
            (torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2
        )

        # Combine all components
        total_loss = total_loss + bce_loss + sparsity_loss + smoothness_loss

    return total_loss / batch_size

def ranking_loss_val(scores, labels, batch_size, lamda_sparsity=8e-5, lamda_smooth=8e-5):
    """
    # Ranking loss with Binary Cross-Entropy for weakly-supervised MIL anomaly detection.
    # Uses video-level Binary Cross-Entropy instead of top-k selection.
"""
    num_segments = scores.shape[0] // batch_size  # Segments per video
    total_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)

    for i in range(batch_size):
        # Extract the scores for this video
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i].float()  # Single video-level label

        # Compute the mean score for the video
        mean_score = video_scores.mean()

        # Binary Cross-Entropy loss for the video
        bce_loss = F.binary_cross_entropy_with_logits(mean_score, video_label)

        # Sparsity and Smoothness
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum(
            (torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2
        )

        # Combine all components
        total_loss = total_loss + bce_loss + sparsity_loss + smoothness_loss

    return total_loss / batch_size



criterion=ranking_loss
criterion_val = ranking_loss_val

from sklearn.preprocessing import StandardScaler
StandarScale = StandardScaler()

def train_epoch(train_loader, model, optimizer, criterion, device, scaler, batch_size):
    """
    #Train the model for one epoch using ranking_loss.
"""
    model.train()  # Set model to training mode
    all_probs = []
    all_labels = []
    total_loss = 0

    for normal_features, anomalous_features in tqdm(train_loader, desc="Training"):
        # Combine features
        if normal_features.dim() > 2:
            normal_features = normal_features.squeeze(-1).squeeze(-1)
            normal_features = normal_features.view(-1, 2048)

            normal_features = torch.tensor(
                    StandarScale.fit_transform(normal_features.cpu().numpy()),  # Scale on CPU
                    dtype=torch.float32,  # Convert back to float
                ).to(device) 
            
        if anomalous_features.dim() > 2:
            anomalous_features = anomalous_features.squeeze(-1).squeeze(-1)
            anomalous_features = anomalous_features.view(-1, 2048)
            anomalous_features = torch.tensor(
                    StandarScale.fit_transform(anomalous_features.cpu().numpy()),  # Scale on CPU
                    dtype=torch.float32,  # Convert back to float
                ).to(device) 

        normal_features = normal_features.unsqueeze(2).to(device)
        anomalous_features = anomalous_features.unsqueeze(2).to(device)
        inputs = torch.cat((normal_features, anomalous_features), dim=0)
        # Labels: 1 for normal, 0 for anomalous
        labels = torch.cat((
            torch.ones(len(normal_features)).to(device),
            torch.zeros(len(anomalous_features)).to(device)
        ), dim=0)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
            outputs = model(inputs)  # [total_snippets, 1]
            loss = criterion(outputs, labels, batch_size)

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Collect outputs and labels for evaluation
        all_probs.extend(torch.sigmoid(outputs).view(-1).detach().cpu().tolist())
        all_labels.extend(labels.view(-1).detach().cpu().tolist())

    # Debugging: Check lengths
    print(f"Length of all_probs: {len(all_probs)}, Length of all_labels: {len(all_labels)}")

    # Ensure consistency before metrics calculation
    assert len(all_probs) == len(all_labels), f"Inconsistent lengths: all_probs={len(all_probs)}, all_labels={len(all_labels)}"

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")

    # Convert to NumPy arrays for sklearn metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Use precision_recall_curve to get thresholds
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    print(f'Thresholds: {thresholds}, Length of thresholds: {len(thresholds)}')

    # Calculate accuracy for each threshold
    accuracy_ls = []
    for thres in thresholds:
        y_pred = np.where(all_probs > thres, 1, 0)
        accuracy_ls.append(accuracy_score(all_labels, y_pred))

    # Find the best threshold based on accuracy
    accuracy_ls = pd.DataFrame({'thresholds': thresholds, 'accuracy': accuracy_ls})
    best_row = accuracy_ls.loc[accuracy_ls['accuracy'].idxmax()]
    best_threshold_accuracy = best_row['thresholds']
    best_accuracy = best_row['accuracy']

    print(f"Best Threshold (Accuracy): {best_threshold_accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve", lw=2)
    plt.axvline(x=best_threshold_accuracy, color='r', linestyle='--', label=f"Best Threshold (Accuracy) = {best_threshold_accuracy:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"train_precision_recall_curve_{epoch+1}.png")
    plt.close()

    # Compute ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"train_roc_curve_{epoch+1}.png")
    plt.close()

    return avg_loss, best_threshold_accuracy, best_accuracy, roc_auc

def validate_epoch(val_loader, model, criterion_val, device, batch_size, epoch):
    """
    #Validate the model for one epoch using ranking_loss.
"""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():  # No gradient calculation
        for normal_features, anomalous_features in tqdm(val_loader, desc="Validation"):
            # Combine features
            if normal_features.dim() > 2:
                normal_features = normal_features.squeeze(-1).squeeze(-1)
                normal_features = normal_features.view(-1, 2048)
                normal_features = torch.tensor(
                    StandarScale.fit_transform(normal_features.cpu().numpy()),  # Scale on CPU
                    dtype=torch.float32
                ).to(device)

            if anomalous_features.dim() > 2:
                anomalous_features = anomalous_features.squeeze(-1).squeeze(-1)
                anomalous_features = anomalous_features.view(-1, 2048)
                anomalous_features = torch.tensor(
                    StandarScale.fit_transform(anomalous_features.cpu().numpy()),  # Scale on CPU
                    dtype=torch.float32
                ).to(device)

            normal_features = normal_features.unsqueeze(2).to(device)
            anomalous_features = anomalous_features.unsqueeze(2).to(device)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)

            # Labels: 1 for normal, 0 for anomalous
            labels = torch.cat((
                torch.ones(len(normal_features)).to(device),
                torch.zeros(len(anomalous_features)).to(device)
            ), dim=0)

            # Forward pass
            outputs = model(inputs)
            loss = criterion_val(outputs, labels, batch_size)
            total_loss += loss.item()

            # Collect probabilities and labels for evaluation
            all_probs.extend(torch.sigmoid(outputs).view(-1).cpu().tolist())
            all_labels.extend(labels.view(-1).cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    print(f'Thresholds: {thresholds}, Length of thresholds: {len(thresholds)}')

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Validation) - Epoch {epoch + 1}")
    plt.legend()
    plt.grid()
    plt.savefig(f"validation_precision_recall_curve_epoch_{epoch + 1}.png")
    plt.close()

    # Compute ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"Validation ROC AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Validation) - Epoch {epoch + 1}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"validation_roc_curve_epoch_{epoch + 1}.png")
    plt.close()

    return avg_loss, roc_auc

def test_epoch(test_loader, model, criterion_val, device, batch_size, epoch):
    """
    #Test the model after training using ranking_loss.
"""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():  # No gradient calculation
        for normal_features, anomalous_features in tqdm(test_loader, desc="Testing"):
            # Combine features
            if normal_features.dim() > 2:
                normal_features = normal_features.squeeze(-1).squeeze(-1)
                normal_features = normal_features.view(-1, 2048)
                normal_features = torch.tensor(
                    StandarScale.fit_transform(normal_features.cpu().numpy()),  # Scale on CPU
                    dtype=torch.float32
                ).to(device)

            if anomalous_features.dim() > 2:
                anomalous_features = anomalous_features.squeeze(-1).squeeze(-1)
                anomalous_features = anomalous_features.view(-1, 2048)
                anomalous_features = torch.tensor(
                    StandarScale.fit_transform(anomalous_features.cpu().numpy()),  # Scale on CPU
                    dtype=torch.float32
                ).to(device)

            normal_features = normal_features.unsqueeze(2).to(device)
            anomalous_features = anomalous_features.unsqueeze(2).to(device)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)

            # Labels: 1 for normal, 0 for anomalous
            labels = torch.cat((
                torch.ones(len(normal_features)).to(device),
                torch.zeros(len(anomalous_features)).to(device)
            ), dim=0)

            # Forward pass
            outputs = model(inputs)
            loss = criterion_val(outputs, labels, batch_size)
            total_loss += loss.item()

            # Collect probabilities and labels for evaluation
            all_probs.extend(torch.sigmoid(outputs).view(-1).cpu().tolist())
            all_labels.extend(labels.view(-1).cpu().tolist())

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    print(f'Thresholds: {thresholds}, Length of thresholds: {len(thresholds)}')

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Test) - Epoch {epoch + 1}")
    plt.legend()
    plt.grid()
    plt.savefig(f"test_precision_recall_curve_epoch_{epoch + 1}.png")
    plt.close()

    # Compute ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # Plot ROC Curve    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Test) - Epoch {epoch + 1}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"test_roc_curve_epoch_{epoch + 1}.png")
    plt.close()

    return avg_loss, roc_auc

class_weights = {0: 2, 1: 0.5}  # Example: Give more weight to class 1 (anomalous)

# Add L2 Regularization to Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # weight_decay is for L2 regularization

# Rebalancing the Dataset (If Needed)
# Adjust the dataloader creation logic to oversample minority classes or undersample majority classes
# Use techniques like SMOTE if your dataset is imbalanced


num_epochs = 20

# Initialize lists to store metrics
train_losses = []
val_losses = []
test_losses = []
roc_aucs_val = []
roc_aucs_test = []
best_test_roc_auc = float("-inf")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # Training phase
        avg_train_loss, best_threshold_accuracy, best_accuracy, roc_auc_train = train_epoch(
            train_loader, model, optimizer, criterion, device, scaler, batch_size
        )
        print(f"Epoch {epoch + 1}: Training Loss = {avg_train_loss:.4f}, Best Threshold (Accuracy) = {best_threshold_accuracy:.4f}, Best Accuracy = {best_accuracy:.4f}, ROC AUC = {roc_auc_train:.4f}")

        # Validation phase
        avg_val_loss, roc_auc_val = validate_epoch(
            val_loader, model, criterion_val, device, batch_size, epoch
        )
        print(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss:.4f}, ROC AUC (Validation) = {roc_auc_val:.4f}")

        # Adjust learning rate
        scheduler.step(avg_val_loss)

        # Optionally: Test phase after validation
        avg_test_loss, roc_auc_test = test_epoch(
            test_loader, model, criterion_val, device, batch_size, epoch
        )
        print(f"Epoch {epoch + 1}: Test Loss = {avg_test_loss:.4f}, ROC AUC (Test) = {roc_auc_test:.4f}")

        if roc_auc_val > best_test_roc_auc:
            best_test_roc_auc = roc_auc_val
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "val_roc_auc": roc_auc_val,
                "val_loss": avg_val_loss,
            }, f"best_model_val_roc_auc_{roc_auc_val:.4f}.pth")
            print(f"Best model saved with test ROC AUC = {roc_auc_val:.4f}")

        # Adjust the learning rate using the scheduler
        scheduler.step(avg_val_loss)



# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=FutureWarning)
#         avg_loss, best_threshold_accuracy, best_accuracy, roc_auc = train_epoch(
#             train_loader, model, optimizer, criterion, device, scaler, batch_size
#         )
#         print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Best Threshold (Accuracy) = {best_threshold_accuracy:.4f}, Best Accuracy = {best_accuracy:.4f}, ROC AUC = {roc_auc:.4f}")
"""

# epoch_training_test.py

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from I3D_split_train_val_test import create_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import warnings
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

random.seed(42)
from Sequential_Model import DeepGRUModel

# -------------------- DEVICE AND MODEL --------------------
device = torch.device("cuda")

model = DeepGRUModel(
    input_dim=2048,
    hidden_dim=256,
    num_layers=16,
    dropout=0.5,
    use_layer_norm=True
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
scaler = GradScaler()

# -------------------- DIRECTORIES --------------------
train_normal_dir = "split_I3D/train/normal"
train_anomalous_dir = "split_I3D/train/anomalous"
val_normal_dir = "split_I3D/val/normal"
val_anomalous_dir = "split_I3D/val/anomalous"
test_normal_dir = "split_I3D/test/normal"
test_anomalous_dir = "split_I3D/test/anomalous"

max_segments_train = 509
max_segments_val = 253
max_segments_test = 281
batch_size = 4

train_loader = create_dataloader(train_normal_dir, train_anomalous_dir, batch_size, max_segments_train)
val_loader = create_dataloader(val_normal_dir, val_anomalous_dir, batch_size, max_segments_val)
test_loader = create_dataloader(test_normal_dir, test_anomalous_dir, batch_size, max_segments_test)

# -------------------- LOSS FUNCTIONS --------------------
def ranking_loss(scores, labels, batch_size, lamda_sparsity=8e-5, lamda_smooth=8e-5, k_ratio=0.2):
    num_segments = scores.shape[0] // batch_size
    total_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)
    for i in range(batch_size):
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i].float()
        feature_magnitudes = torch.norm(video_scores, p=2, dim=-1)
        k = max(1, int(len(feature_magnitudes) * k_ratio)) if feature_magnitudes.dim() != 0 else 1
        top_k_features, _ = torch.topk(feature_magnitudes, k)
        mean_score = top_k_features.mean()
        bce_loss = F.binary_cross_entropy_with_logits(mean_score, video_label)
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum((torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2)
        total_loss = total_loss + bce_loss + sparsity_loss + smoothness_loss
    return total_loss / batch_size

def ranking_loss_val(scores, labels, batch_size, lamda_sparsity=8e-5, lamda_smooth=8e-5):
    num_segments = scores.shape[0] // batch_size
    total_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)
    for i in range(batch_size):
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i].float()
        mean_score = video_scores.mean()
        bce_loss = F.binary_cross_entropy_with_logits(mean_score, video_label)
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum((torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2)
        total_loss = total_loss + bce_loss + sparsity_loss + smoothness_loss
    return total_loss / batch_size

criterion = ranking_loss
criterion_val = ranking_loss_val

# -------------------- VALIDATION FUNCTION --------------------
def validate_epoch(val_loader, model, criterion_val, device, batch_size, epoch):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for normal_features, anomalous_features in tqdm(val_loader, desc="Validation"):
            def preprocess(x):
                if x.dim() > 2:
                    x = x.squeeze(-1).squeeze(-1).view(-1, 2048)
                    x = torch.tensor(StandardScaler().fit_transform(x.cpu().numpy()), dtype=torch.float32).to(device)
                    return x.unsqueeze(2)
                return x.to(device)
            normal_features = preprocess(normal_features)
            anomalous_features = preprocess(anomalous_features)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)
            labels = torch.cat((torch.ones(len(normal_features)).to(device), torch.zeros(len(anomalous_features)).to(device)), dim=0)
            outputs = model(inputs)
            loss = criterion_val(outputs, labels, batch_size)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(outputs).view(-1).cpu().tolist())
            all_labels.extend(labels.view(-1).cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot and save PR and ROC curves
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Validation Precision-Recall Epoch {epoch+1}")
    plt.savefig(f"validation_precision_recall_curve_epoch_{epoch+1}.png")
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"Validation ROC Curve Epoch {epoch+1}")
    plt.savefig(f"validation_roc_curve_epoch_{epoch+1}.png")
    plt.close()

    return avg_loss, roc_auc

# -------------------- TEST FUNCTION --------------------
def test_epoch(test_loader, model, criterion_val, device, batch_size, epoch):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for normal_features, anomalous_features in tqdm(test_loader, desc="Testing"):
            def preprocess(x):
                if x.dim() > 2:
                    x = x.squeeze(-1).squeeze(-1).view(-1, 2048)
                    x = torch.tensor(StandardScaler().fit_transform(x.cpu().numpy()), dtype=torch.float32).to(device)
                    return x.unsqueeze(2)
                return x.to(device)
            normal_features = preprocess(normal_features)
            anomalous_features = preprocess(anomalous_features)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)
            labels = torch.cat((torch.ones(len(normal_features)).to(device), torch.zeros(len(anomalous_features)).to(device)), dim=0)
            outputs = model(inputs)
            loss = criterion_val(outputs, labels, batch_size)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(outputs).view(-1).cpu().tolist())
            all_labels.extend(labels.view(-1).cpu().tolist())

    avg_loss = total_loss / len(test_loader)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot and save PR and ROC curves
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Test Precision-Recall Epoch {epoch+1}")
    plt.savefig(f"test_precision_recall_curve_epoch_{epoch+1}.png")
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"Test ROC Curve Epoch {epoch+1}")
    plt.savefig(f"test_roc_curve_epoch_{epoch+1}.png")
    plt.close()

    return avg_loss, roc_auc

# -------------------- TRAINING LOOP --------------------
num_epochs = 20
train_losses = []
val_losses = []
test_losses = []
roc_aucs_val = []
roc_aucs_test = []
best_test_roc_auc = float("-inf")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        # -------------------- TRAIN --------------------
        model.train()
        all_probs, all_labels, total_loss = [], [], 0
        for normal_features, anomalous_features in tqdm(train_loader, desc="Training"):
            def preprocess(x):
                if x.dim() > 2:
                    x = x.squeeze(-1).squeeze(-1).view(-1, 2048)
                    x = torch.tensor(StandardScaler().fit_transform(x.cpu().numpy()), dtype=torch.float32).to(device)
                    return x.unsqueeze(2)
                return x.to(device)
            normal_features = preprocess(normal_features)
            anomalous_features = preprocess(anomalous_features)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)
            labels = torch.cat((torch.ones(len(normal_features)).to(device), torch.zeros(len(anomalous_features)).to(device)), dim=0)
            optimizer.zero_grad()
            with autocast("cuda", enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels, batch_size)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(outputs).view(-1).detach().cpu().tolist())
            all_labels.extend(labels.view(-1).detach().cpu().tolist())

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        all_probs, all_labels = np.array(all_probs), np.array(all_labels)
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        accuracy_ls = [accuracy_score(all_labels, np.where(all_probs > thres, 1, 0)) for thres in thresholds]
        best_idx = np.argmax(accuracy_ls)
        best_threshold_accuracy, best_accuracy = thresholds[best_idx], accuracy_ls[best_idx]
        print(f"Best Threshold (Accuracy): {best_threshold_accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc_train = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc_train:.4f}")

        # -------------------- VALIDATION --------------------
        avg_val_loss, roc_auc_val = validate_epoch(val_loader, model, criterion_val, device, batch_size, epoch)
        print(f"Validation Loss: {avg_val_loss:.4f}, ROC AUC (Validation): {roc_auc_val:.4f}")

        scheduler.step(avg_val_loss)
        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

        # -------------------- TEST --------------------
        avg_test_loss, roc_auc_test = test_epoch(test_loader, model, criterion_val, device, batch_size, epoch)
        print(f"Test Loss: {avg_test_loss:.4f}, ROC AUC (Test): {roc_auc_test:.4f}")

        if roc_auc_val > best_test_roc_auc:
            best_test_roc_auc = roc_auc_val
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_roc_auc": roc_auc_val,
                "val_loss": avg_val_loss,
            }, f"best_model_val_roc_auc_{roc_auc_val:.4f}.pth")
            print(f"Best model saved with val ROC AUC = {roc_auc_val:.4f}")

if __name__=="__main__":
    
    import torch
    from extract_segments import extract_segments_from_video  # Function to divide videos into segments
    from extract_features import extract_features  # Function to extract features from segments
    from Sequential_Model import DeepGRUModel  # <<< USE DeepGRUModel
    from sklearn.preprocessing import StandardScaler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer_video(video_path, model, device, best_threshold=0.6):
        """
        Perform inference on a single video to predict its label (0 or 1).

        Args:
            video_path (str): Path to the video file.
            model (torch.nn.Module): Trained MIL model (DeepGRUModel).
            device (torch.device): Device for computation (CPU or GPU).
            best_threshold (float): Best threshold obtained during training.

        Returns:
            tuple: (predicted_probability, predicted_label)
        """
        try:
            # Step 1: Extract video segments
            segments = extract_segments_from_video(video_path, segment_size=16, target_shape=(320, 320), frame_skip=5)
            if not segments:
                raise ValueError("No segments were extracted from the video.")

            # Step 2: Extract features for each segment
            features = extract_features(segments, device)  # List of [feature_dim] tensors
            features = torch.stack(features).squeeze(-1).squeeze(-1).squeeze(-1)  # [num_segments, feature_dim]
            features = features.to(device)

            # Optional: Apply StandardScaler if used during training
            scaler = StandardScaler()
            features_np = features.cpu().numpy()
            features_np = scaler.fit_transform(features_np)  # Standardize
            features = torch.tensor(features_np, dtype=torch.float32).to(device)

            # Step 3: Predict using the trained MIL model
            model.eval()
            
            with torch.no_grad():
                features = features.unsqueeze(0)  # [1, num_segments, feature_dim] for GRU
                scores = model(features)          # Output shape [1, num_segments, 1]
                scores = scores.squeeze(0).squeeze(-1)  # [num_segments]

                print(f"torch.mean(scores): {torch.mean(scores).item():.4f}")
                print(f"torch.max(scores): {torch.max(scores).item():.4f}")

                predicted_probability = torch.mean(scores).item()

            predicted_label = 0 if predicted_probability > best_threshold else 1

            return predicted_probability, predicted_label

        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None

    # File path for the video
    video_path = r"E:\MIL\Code\videos\walking.mp4"  # <<< Change to your video

    # Load the trained MIL model
    model = DeepGRUModel(
        input_dim=2048,
        hidden_dim=256,
        num_layers=5,
        dropout=0.5,
        use_layer_norm=False
    ).to(device)

    try:
        checkpoint = torch.load("trained_model.pth", map_location=device)  # <<< Adjust this path
        model.load_state_dict(checkpoint['model_state'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading the model: {e}")

    # Set best threshold
    best_threshold = 0.713  # <<< Replace with your training threshold

    predicted_probability, predicted_label = infer_video(video_path, model, device, best_threshold)

    if predicted_probability is not None:
        print(f"Predicted Probability of being Anomalous: {predicted_probability:.4f}")
        print(f"Best Threshold Used: {best_threshold}")
        print(f"Predicted Label: {'Anomalous' if predicted_label == 0 else 'Normal'}")