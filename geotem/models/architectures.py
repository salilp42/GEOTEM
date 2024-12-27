"""
Model architectures for ECG classification.
Author: Salil Patel
"""

import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """Cross-attention block for feature interaction."""
    def __init__(self, d_model=64, nhead=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 4, d_model)
        Returns:
            Output tensor of shape (B, 4, d_model)
        """
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x

class GTN_CrossAttention(nn.Module):
    """Geometric-Temporal Network with cross-attention."""
    def __init__(self, N=96, out_channels=32, d_model=64, nhead=4, dropout=0.2):
        super().__init__()
        self.N = N
        self.out_channels = out_channels
        self.d_model = d_model

        # CNN branches (one per channel)
        self.branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            )
            for _ in range(4)
        ])
        
        # Projection layers
        self.proj = nn.ModuleList([
            nn.Linear(out_channels * (N//4), d_model)
            for _ in range(4)
        ])

        # Cross-attention
        self.cross_attn = CrossAttentionBlock(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 4*N)
        Returns:
            Binary classification probability
        """
        B = x.shape[0]
        x_reshaped = x.view(B, 4, self.N)
        
        token_list = []
        for i in range(4):
            ch_ = x_reshaped[:, i, :].unsqueeze(1)
            out_ = self.branch[i](ch_)
            out_ = out_.flatten(start_dim=1)
            out_ = self.proj[i](out_)
            token_list.append(out_)
        
        tokens = torch.stack(token_list, dim=1)
        tokens = self.cross_attn(tokens)
        pooled = tokens.mean(dim=1)
        out = self.classifier(pooled)
        return out

class BaselineCNN(nn.Module):
    """1D CNN baseline for ECG classification."""
    def __init__(self, sequence_length=96):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64*(sequence_length//4), 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        out = torch.sigmoid(self.fc2(x))
        return out

class BaselineLSTM(nn.Module):
    """Bidirectional LSTM baseline for ECG classification."""
    def __init__(self, sequence_length=96, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_size,
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size*2, 1)
    
    def forward(self, x):
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        out = torch.sigmoid(self.fc(x))
        return out
