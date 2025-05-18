
import torch
import torch.nn as nn
from torch.nn import functional as F

class DeepGRUModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=5, dropout=0.5, use_layer_norm=False):
        super(DeepGRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.fc = nn.Linear(hidden_dim, 1)  # Final classification layer
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        """
        Forward pass for Deep GRU Model
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        x = x.float()
        if x.dim() > 2:
            x = x.squeeze(-1).squeeze(-1).squeeze(-1)
            x = x.view(-1, x.size(-1))
            
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_length, hidden_dim]

        if self.use_layer_norm:
            gru_out = self.layer_norm(gru_out)
       
        out = self.fc(gru_out)  
        return self.sigmoid(out)

class SequentialMILModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        """
        Initializes a Sequential MIL Model for anomaly detection.

        Parameters:
        - input_dim (int): Input feature dimension (default: 2048).
        - hidden_dim (int): Hidden layer dimension (default: 512).
        """
        super(SequentialMILModel, self).__init__()
        print("SequentialMILModel Initialized")
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)         # Second fully connected layer
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)                 # Final output layer
        self.bn3 = nn.BatchNorm1d(1)
        
        self.dropout = nn.Dropout(0.6)              # Dropout with probability 0.6
        self.relu = nn.ReLU()                       # ReLU activation
        self.sigmoid = nn.Sigmoid()                 # Sigmoid activation for output

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
        - x (torch.Tensor): Input features of shape [num_segments, input_dim].

        Returns:
        - torch.Tensor: Raw logits for each segment. Shape: [num_segments, 1].
        """
        x = x.float()
        if x.dim() > 2:  # Reshape tensor if needed
            x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
            x = x.view(-1, x.size(-1))

        # Fully connected layers with dropout and ReLU
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.bn2(x)

        # Final output layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.sigmoid(x)

        return x.squeeze(-1)  # Return anomaly score for each segment

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load features
    features = torch.load(r'E:\MIL\Code\normal_features\Normal_Videos436_x264_features.pt')['features'].to(device)

    print(f"Input features shape: {features.shape}")
    
    # Initialize model
    model = DeepGRUModel(
            input_dim=2048,  # Feature size
            hidden_dim=256,  # GRU hidden size
            num_layers=16,   # Number of GRU layers
            dropout=0.5,     # Dropout rate
            use_layer_norm=True  # Use layer normalization
        ).to(device)
    model.eval()
    # Forward pass
    with torch.no_grad():
        feature= features.unsqueeze(0).to(device)
        print(f'feature shape:{feature.shape}')
        output = model(features)
    print(f"Output shape: {output.shape}")
    print(f'output:{output}')
    max = 0
    for i in output:
        if i > max:
            max = i
    print(f'max:{max}')
    # print(f"Output scores: {output}")
    # print(f'Max of output:{output.max()}')
    print(f'length of features : {len(features)}')
    print(f'length of output:  {len(output)}')


