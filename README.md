# GEOTEM: Geometric and Temporal ECG Models

A comprehensive framework for ECG classification that combines geometric theory, topological data analysis (TDA), and deep learning. This repository implements and compares three approaches:

- **GTN**: A novel Geometric-Temporal Network that uses cross-attention among multiple geometric and topological signal representations
- **CNN**: Baseline 1D CNN with Grad-CAM visualization
- **LSTM**: Baseline LSTM with Integrated Gradients

## Background & Novel Aspects

### Geometric Signal Processing
ECG signals naturally exist on a geometric manifold where important features are invariant under certain transformations. Our approach leverages three key geometric aspects:

1. **Discrete Curvature**: Captures the local geometry of the signal through second-order variations, particularly effective for detecting P-waves and T-waves.

2. **Geometric Phase**: Represents the signal's dynamic evolution in phase space, making it particularly sensitive to rhythm abnormalities and morphological changes.

3. **Topological Features**: Uses persistent homology to capture multi-scale topological features, robust to noise and amplitude variations.

### Cross-Attention Mechanism
The GTN model introduces a novel cross-attention mechanism that allows different geometric representations to interact:

```
Raw Signal ─┐
            │
Curvature ──┼─► Cross-Attention ─► Classification
            │
Phase ──────┤
            │
TDA Features┘
```

This architecture enables the model to:
- Learn which geometric features are most relevant for different ECG patterns
- Combine local and global geometric information
- Maintain interpretability through attention weights

## Features

- **Multi-scale Analysis**: Combines features at different geometric scales
- **Interpretability**: 
  - GTN: Cross-attention visualization
  - CNN: Grad-CAM for temporal importance
  - LSTM: Integrated Gradients for feature attribution
- **Robust Evaluation**:
  - 5-fold cross-validation
  - Statistical significance testing
  - Comprehensive metrics suite
- **Visualization**:
  - ROC curves with confidence intervals
  - Feature importance overlays
  - Confusion matrices
  - Metric comparisons

## Installation

```bash
# Clone the repository
git clone https://github.com/salilp42/GEOTEM.git
cd GEOTEM

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
GEOTEM/
├── data/                    # Data directory
│   └── ECG200/             # ECG200 dataset
├── geotem/                 # Main package
│   ├── models/            # Model architectures
│   │   └── architectures.py  # GTN, CNN, LSTM implementations
│   ├── features/          # Feature extraction
│   │   └── geometric_features.py  # Geometric feature computation
│   ├── visualization/     # Plotting utilities
│   │   └── interpretability.py    # Model interpretation tools
│   └── utils/            # Helper functions
├── notebooks/             # Jupyter notebooks
├── scripts/              # Training scripts
│   └── train.py         # Main training pipeline
├── tests/               # Unit tests
└── results/            # Output directory
```

## Usage

### Basic Usage
```python
from geotem.models import GTN_CrossAttention
from geotem.features import AdvancedGeometricFeatures

# Extract geometric features
feature_extractor = AdvancedGeometricFeatures()
X_geo = feature_extractor.extract_features(X_raw)

# Initialize and train GTN model
model = GTN_CrossAttention(
    N=sequence_length,
    out_channels=32,
    d_model=64,
    nhead=4
)
```

### Training Pipeline
```bash
# Run full training pipeline with cross-validation
python scripts/train.py
```

## Model Architecture Details

### GTN (Geometric-Temporal Network)
- **Input**: 4-channel representation (raw, curvature, phase, TDA)
- **Feature Processing**: Independent CNN branches per channel
- **Feature Interaction**: Multi-head cross-attention
- **Output**: Binary classification through MLP

### Baseline Models
- **CNN**: 1D convolutions with max pooling
- **LSTM**: Bidirectional LSTM with final state classification

## Results

### Hold-out Test Set Performance

Performance metrics on the ECG200 test set:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | MCC |
|-------|----------|-----------|--------|----------|---------|-----|
| GTN   | 0.8700   | 0.8806    | 0.9219 | 0.9008   | 0.9158  | 0.7142 |
| CNN   | 0.8300   | 0.8730    | 0.8594 | 0.8661   | 0.9188  | 0.6335 |
| LSTM  | 0.6600   | 0.6829    | 0.8750 | 0.7671   | 0.7214  | 0.1909 |

### Statistical Analysis

Pairwise Wilcoxon tests on 5-fold cross-validation metrics to assess statistical significance:

#### GTN vs CNN
- Accuracy: p=0.0679 
- Precision: p=0.0625 
- Recall: p=0.4142
- F1-score: p=0.1250
- AUC-ROC: p=0.4375
- MCC: p=0.0625 

#### GTN vs LSTM
- Accuracy: p=0.1250
- Precision: p=0.1250
- Recall: p=0.4142
- F1-score: p=0.1250
- AUC-ROC: p=0.1250
- MCC: p=0.1250

#### CNN vs LSTM
- Accuracy: p=1.0000
- Precision: p=0.3125
- Recall: p=0.1441
- F1-score: p=0.8125
- AUC-ROC: p=0.3125
- MCC: p=0.6250

### Key Findings

1. **GTN Performance**: 
   - Best overall performance on test set (Accuracy: 87%, F1: 0.90)
   - Highest precision (0.88) and recall (0.92)
   - Strong MCC score (0.71) indicating reliable predictions

2. **Model Comparison**:
   - GTN shows marginally significant improvements over CNN in accuracy, precision, and MCC
   - Both GTN and CNN substantially outperform LSTM
   - High AUC-ROC scores for GTN (0.92) and CNN (0.92) suggest excellent discrimination ability

3. **Statistical Significance**:
   - Most pronounced differences between GTN and CNN (p < 0.07 for key metrics)
   - Less significant differences between CNN and LSTM
   - Results suggest GTN's geometric features provide meaningful improvements

*Note: Statistical tests performed on 5-fold CV results; p-values < 0.05 considered significant, < 0.10 marginally significant*

## Citation

If you use this code in your research, please cite:

```bibtex
@software{patel2024geotem,
  author = {Patel, Salil},
  title = {GEOTEM: Geometric and Temporal ECG Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/salilp42/GEOTEM}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of particular interest:

- Additional geometric features
- New attention mechanisms
- Performance optimizations
- Extended documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Salil Patel**
