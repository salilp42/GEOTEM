"""
Interpretability tools for ECG models.
Author: Salil Patel
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class GradCAM1D:
    """1D Grad-CAM implementation for CNN interpretability."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, inp, out):
            self.activations = out
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        self.model.conv2.register_forward_hook(forward_hook)
        self.model.conv2.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor):
        """Generate class activation map for input tensor."""
        pred = self.model(input_tensor)
        self.model.zero_grad()
        pred.backward()
        
        alpha = torch.mean(self.gradients, dim=(0,2), keepdim=True)
        cam = self.activations * alpha
        cam = torch.mean(cam, dim=1).detach().cpu().numpy().squeeze()
        
        # Interpolate to original length
        length = input_tensor.shape[-1]
        x_ = np.arange(cam.shape[0])
        x_new = np.linspace(0, cam.shape[0]-1, length)
        f_int = interp1d(x_, cam, kind='linear')
        cam_ups = f_int(x_new)
        
        # Normalize
        cam_ups = np.maximum(cam_ups, 0)
        if cam_ups.max() > 1e-8:
            cam_ups /= cam_ups.max()
        return cam_ups

def integrated_gradients(model, input_tensor, steps=50):
    """
    Compute Integrated Gradients attribution.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        steps: Number of steps for path integral
        
    Returns:
        Attribution scores
    """
    device = next(model.parameters()).device
    model.eval()
    
    x_clone = input_tensor.detach().clone()
    x_clone.requires_grad_(True)
    baseline = torch.zeros_like(x_clone)
    total_grads = torch.zeros_like(x_clone)
    
    for alpha in np.linspace(0, 1, steps+1):
        interp = baseline + alpha*(x_clone - baseline)
        interp = interp.clone().detach().requires_grad_(True)
        
        out = model(interp)
        out.backward(torch.ones_like(out))
        
        total_grads += interp.grad.data
    
    avg_grads = total_grads / (steps+1)
    ig = (x_clone - baseline)*avg_grads
    return ig.detach()

def plot_gtn_time_attention(model, X_test_geo, y_test, output_dir,
                          n_samples=3, seq_len=96):
    """Plot GTN's gradient-based importance on raw signal."""
    model.eval()
    device = next(model.parameters()).device
    idxs = np.random.choice(len(X_test_geo), size=n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(10,3*n_samples), dpi=300)
    if n_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        idx_ = idxs[i]
        x_ = torch.FloatTensor(X_test_geo[idx_]).unsqueeze(0).to(device)
        label_ = y_test[idx_]

        x_.requires_grad = True
        pred = model(x_)
        pred_val = pred.item()
        pred.backward()
        grad_np = x_.grad.detach().cpu().numpy().squeeze()
        grad_raw = np.abs(grad_np[:seq_len])
        if grad_raw.max() > 1e-8:
            grad_raw /= grad_raw.max()

        raw_signal = X_test_geo[idx_][:seq_len]
        ax.plot(raw_signal, color='black')
        ax2 = ax.twinx()
        ax2.fill_between(np.arange(seq_len), grad_raw, color='blue', alpha=0.3)
        ax2.set_ylim(0,1.05)
        ax.set_title(f"GTN Cross-Attn idx={idx_}, label={label_}, pred={pred_val:.3f}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "gtn_time_attention_overlay.png", dpi=300)
    plt.close()

def plot_cnn_gradcam(model, X_test, y_test, output_dir, n_samples=3):
    """Plot CNN Grad-CAM visualizations."""
    device = next(model.parameters()).device
    model.eval()
    idxs = np.random.choice(len(X_test), size=n_samples, replace=False)
    
    cam_extractor = GradCAM1D(model)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(10,3*n_samples), dpi=300)
    if n_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        idx_ = idxs[i]
        x_ = torch.FloatTensor(X_test[idx_]).unsqueeze(0).to(device)
        pred_val = model(x_).item()
        label_ = y_test[idx_]
        
        cam_ = cam_extractor.generate_cam(x_)
        ax.plot(X_test[idx_], color='black')
        ax2 = ax.twinx()
        ax2.fill_between(range(len(X_test[idx_])), cam_, color='red', alpha=0.4)
        ax2.set_ylim(0,1.05)
        ax.set_title(f"CNN Grad-CAM idx={idx_}, label={label_}, pred={pred_val:.3f}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_gradcam.png", dpi=300)
    plt.close()

def plot_lstm_integrated_gradients(model, X_test, y_test, output_dir, n_samples=3):
    """Plot LSTM Integrated Gradients visualizations."""
    device = next(model.parameters()).device
    model.eval()
    idxs = np.random.choice(len(X_test), size=n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(10,3*n_samples), dpi=300)
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        idx_ = idxs[i]
        x_ = torch.FloatTensor(X_test[idx_]).unsqueeze(0).to(device)
        pred_val = model(x_).item()
        label_ = y_test[idx_]

        ig_ = integrated_gradients(model, x_, steps=50).cpu().numpy().squeeze()
        ig_pos = np.clip(ig_, 0, None)
        ig_neg = np.clip(ig_, None, 0)
        m_ = max(ig_pos.max(), abs(ig_neg.min()), 1e-8)
        ig_pos /= m_
        ig_neg /= -m_

        ax.plot(X_test[idx_], color='black')
        ax2 = ax.twinx()
        ax2.fill_between(range(len(X_test[idx_])), 0, ig_pos, color='red', alpha=0.3)
        ax2.fill_between(range(len(X_test[idx_])), 0, -ig_neg, color='blue', alpha=0.3)
        ax2.set_ylim([-1,1])
        ax.set_title(f"LSTM IG idx={idx_}, label={label_}, pred={pred_val:.3f}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "lstm_integrated_gradients.png", dpi=300)
    plt.close()
