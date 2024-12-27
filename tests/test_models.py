"""
Unit tests for model architectures.
Author: Salil Patel
"""

import torch
import pytest
from geotem.models.architectures import (
    GTN_CrossAttention,
    BaselineCNN,
    BaselineLSTM
)

@pytest.fixture
def sample_batch():
    batch_size = 4
    seq_len = 96
    X = torch.randn(batch_size, seq_len)
    y = torch.randint(0, 2, (batch_size,)).float()
    return X, y

def test_gtn_forward(sample_batch):
    X, y = sample_batch
    batch_size = X.shape[0]
    seq_len = X.shape[1]
    
    # Create 4-channel input
    X_4ch = torch.cat([X, X, X, X], dim=1)  # (B, 4*seq_len)
    
    model = GTN_CrossAttention(N=seq_len)
    out = model(X_4ch)
    
    assert out.shape == (batch_size, 1)
    assert torch.all((out >= 0) & (out <= 1))

def test_cnn_forward(sample_batch):
    X, y = sample_batch
    batch_size = X.shape[0]
    seq_len = X.shape[1]
    
    model = BaselineCNN(sequence_length=seq_len)
    out = model(X)
    
    assert out.shape == (batch_size, 1)
    assert torch.all((out >= 0) & (out <= 1))

def test_lstm_forward(sample_batch):
    X, y = sample_batch
    batch_size = X.shape[0]
    seq_len = X.shape[1]
    
    model = BaselineLSTM(sequence_length=seq_len)
    out = model(X)
    
    assert out.shape == (batch_size, 1)
    assert torch.all((out >= 0) & (out <= 1))
