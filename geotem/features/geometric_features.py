"""
Advanced geometric and topological feature extraction for ECG signals.
Author: Salil Patel
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from ripser import ripser
    HAVE_RIPSER = True
except ImportError:
    HAVE_RIPSER = False
    print("Warning: ripser not installed. TDA features will be zero vectors.")

class AdvancedGeometricFeatures:
    """
    Extracts geometric and topological features from ECG signals:
        - Raw signal
        - Discrete curvature
        - Geometric phase
        - TDA-based channel (if ripser available)
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def compute_curvature(self, x):
        """Compute discrete curvature of the signal."""
        dx = np.gradient(x)
        ddx = np.gradient(dx)
        curvature = np.abs(ddx) / (1 + dx**2)**1.5
        return curvature
    
    def compute_phase(self, x):
        """Compute geometric phase of the signal."""
        return np.arctan2(np.gradient(x), x + 1e-8)

    def compute_tda_channel(self, x):
        """
        Compute topological features using persistent homology.
        Falls back to zero vector if ripser not available.
        """
        if not HAVE_RIPSER:
            return np.zeros_like(x)
            
        data_1d = x.reshape(-1, 1)
        rips_dict = ripser(data_1d, maxdim=1)
        dgms = rips_dict['dgms']
        
        all_pairs = []
        for dgm in dgms:
            for (birth, death) in dgm:
                if death == np.inf:
                    death = len(x)
                if death < birth:
                    continue
                lifetime = death - birth
                all_pairs.append(lifetime)
                
        if len(all_pairs) == 0:
            return np.zeros_like(x)
            
        max_life = max(all_pairs)
        if max_life < 1e-8:
            return np.zeros_like(x)
            
        hist_, bin_edges = np.histogram(
            all_pairs, 
            bins=len(x), 
            range=(0, max_life)
        )
        hist_ = hist_.astype(float)
        if hist_.max() > 0:
            hist_ /= hist_.max()
            
        return hist_

    def extract_features(self, X, fit=True):
        """
        Extract all features from a batch of signals.
        
        Args:
            X: Array of shape (n_samples, sequence_length)
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Array of shape (n_samples, 4*sequence_length)
        """
        feats = []
        for i in range(X.shape[0]):
            ts = X[i]
            N = len(ts)
            curv = self.compute_curvature(ts)
            phs  = self.compute_phase(ts)
            tda_ = self.compute_tda_channel(ts)
            combined = np.concatenate([ts, curv, phs, tda_])
            feats.append(combined)
            
        feats = np.array(feats)
        if fit:
            feats = self.scaler.fit_transform(feats)
        else:
            feats = self.scaler.transform(feats)
            
        return feats
