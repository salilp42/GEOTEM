# GEOTEM: GEOmetric Time-series Evaluation through Morphology

A deep learning framework using differential geometry and topological data analysis (TDA) for time series classification. GEOTEM implements principled geometric feature extraction combined with modern transformer architectures.

## Geometric Foundations

GEOTEM builds on established geometric analysis techniques:

- **Differential Geometry**: Utilizes discrete curvature and geometric phase analysis to capture local shape dynamics
- **Topological Features**: Incorporates persistence diagrams from TDA to capture global topological structure
- **Morphological Analysis**: Extracts shape-based features using proven signal processing techniques

## Core Architecture

The main GEOTEM architecture consists of:

1. **Multi-channel Geometric Feature Extraction**
   - Raw signal analysis preserving original information
   - Discrete curvature computation capturing local geometry
   - Geometric phase extraction for dynamic features
   - TDA persistence features encoding topological information

2. **Cross-attention Mechanism**
   - Novel transformer-based attention between geometric channels
   - Learned feature interactions guided by data geometry
   - Interpretable attention weights for feature importance

## Requirements

```bash
numpy
pandas
torch
scikit-learn
matplotlib
seaborn
ripser  # For TDA features
```

## Usage

```python
from geotem.features import GeometricFeatures
from geotem.models import GeometricTransformer

# Extract geometric features
geo_features = GeometricFeatures()
X_transformed = geo_features.transform(X_timeseries)

# Initialize model
model = GeometricTransformer(
    n_channels=4,          # Number of geometric feature channels
    d_model=64,           # Transformer dimension
    n_heads=4             # Number of attention heads
)

# Train model
model.fit(X_transformed, y)
```

## Performance Highlights

On benchmark time series datasets, GEOTEM demonstrates:
- Improved classification accuracy through geometric feature integration
- Enhanced robustness to noise and deformation
- Interpretable feature importance through attention visualization

## Validation & Reproducibility

The repository includes:
- 5-fold cross-validation pipeline
- Statistical significance testing
- Comprehensive visualization tools
- Detailed logging of experiments

## Citation

```bibtex
@article{geotem2024,
  title={GEOTEM: Geometric Time-series Evaluation through Morphology},
  author={[Salil Patel]},
  year={2024}
}
```

## License

MIT License

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

Key areas for contribution:
- Additional geometric feature extractors
- New transformer architectures
- Performance optimizations
- Documentation improvements

Please ensure all contributions maintain the project's focus on geometric analysis and include appropriate tests.