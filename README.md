# anamoly-detection-in-heathcare-data using hybrid model
## Overview

This repository contains an advanced machine learning algorithm specifically designed for the CICIoT2023 dataset, featuring an Enhanced Gazelle Optimization Algorithm (GOA) for hyperparameter optimization and a hybrid ensemble model combining Random Forest, XGBoost, and LightGBM classifiers.

## Key Features

### 🚀 Advanced Optimization
- **Enhanced Gazelle Optimization Algorithm (GOA)** with adaptive mechanisms
- Levy flight random walk for better exploration
- Population diversity maintenance
- Cross-validation based fitness evaluation

### 🤖 Hybrid Ensemble Model
- **Random Forest** with balanced class weights
- **XGBoost** with scale-aware parameters
- **LightGBM** for efficient gradient boosting
- Weighted ensemble predictions for optimal performance

### 📊 Comprehensive Data Processing
- Advanced preprocessing with feature engineering
- Intelligent outlier handling using IQR method
- Multiple class representation support (2-class, 8-class, 34-class)
- Enhanced class balancing with SMOTE integration

### 🔧 Feature Engineering
- Automatic interaction feature creation
- Ratio-based feature generation
- Robust scaling for numerical stability
- Correlation-based feature selection

## Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm
```

### Required Packages
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
xgboost>=1.5.0
lightgbm>=3.3.0
```

## Usage

### Basic Usage

```python
from enhanced_ciciot_algorithm import EnhancedCICIoT2023MLAlgorithm

# Initialize the algorithm
ml_algorithm = EnhancedCICIoT2023MLAlgorithm()

# Run for different class representations
results_2class = ml_algorithm.run_enhanced_algorithm('path_to_dataset.csv', '2-class')
results_8class = ml_algorithm.run_enhanced_algorithm('path_to_dataset.csv', '8-class')
results_34class = ml_algorithm.run_enhanced_algorithm('path_to_dataset.csv', '34-class')
```

### Advanced Usage

```python
# Custom GOA parameters
goa = AdvancedGazelleOptimizationAlgorithm(
    population_size=50,
    max_iterations=150,
    dim=12
)

# Load and preprocess data manually
df = ml_algorithm.load_and_preprocess_data('dataset.csv')
df_balanced = ml_algorithm.enhanced_class_balancing(df, '2-class')
```

## Class Representations

### 2-Class Classification
- **Normal**: Legitimate network traffic
- **Attack**: All types of malicious activities

### 8-Class Classification
1. **Normal**: Legitimate traffic
2. **DDoS**: Distributed Denial of Service attacks
3. **DoS**: Denial of Service attacks
4. **Reconnaissance**: Network scanning and probing
5. **Web_Attack**: Web-based attacks (SQL injection, XSS, etc.)
6. **Brute_Force**: Password and authentication attacks
7. **Spoofing**: Identity spoofing and MITM attacks
8. **Botnet**: Botnet-related activities

### 34-Class Classification
- All original attack categories from the CICIoT2023 dataset
- Maintains fine-grained attack classification

## Algorithm Components

### 1. Enhanced Gazelle Optimization Algorithm (GOA)

The GOA is a bio-inspired optimization algorithm that mimics the social behavior of gazelles. Our enhanced version includes:

- **Adaptive Exploration**: Dynamic adjustment of exploration vs exploitation
- **Levy Flight**: Random walk pattern for better global search
- **Diversity Maintenance**: Prevents premature convergence
- **Cross-Validation Fitness**: Robust evaluation using stratified k-fold

### 2. Advanced Preprocessing Pipeline

```python
# Preprocessing steps include:
- Duplicate removal
- Intelligent missing value handling
- IQR-based outlier capping
- Feature interaction generation
- Robust scaling
```

### 3. Enhanced Class Balancing

```python
# Balancing techniques:
- Intelligent class mapping
- Adaptive sampling strategies
- SMOTE integration
- Stratified sampling
```

### 4. Hybrid Ensemble Model

The ensemble combines three powerful algorithms:

- **Random Forest**: Provides stability and feature importance
- **XGBoost**: Offers gradient boosting efficiency
- **LightGBM**: Delivers fast training and prediction

## Performance Metrics

The algorithm evaluates performance using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Harmonic mean of precision and recall
- **Per-Class Metrics**: Detailed classification report

## Hyperparameter Optimization

The GOA optimizes the following hyperparameters:

### Random Forest
- `n_estimators`: Number of trees (100-300)
- `max_depth`: Maximum tree depth (5-25)
- `min_samples_split`: Minimum samples to split (2-10)
- `min_samples_leaf`: Minimum samples in leaf (1-5)

### XGBoost
- `n_estimators`: Number of boosting rounds (100-500)
- `max_depth`: Maximum tree depth (3-15)
- `learning_rate`: Step size shrinkage (0.01-0.3)
- `subsample`: Subsample ratio (0.6-1.0)
- `colsample_bytree`: Column sampling ratio (0.6-1.0)

### LightGBM
- `n_estimators`: Number of boosting rounds (50-300)
- `learning_rate`: Learning rate (0.01-0.3)

### Ensemble Weights
- Dynamic weight optimization for optimal model combination

## File Structure

```
├── enhanced_ciciot_algorithm.py    # Main algorithm implementation
├── README.md                       # This file
├── requirements.txt               # Required packages
└── examples/                      # Usage examples
    ├── basic_usage.py
    ├── advanced_usage.py
    └── custom_optimization.py
```

## Performance Expectations

### Typical Results (may vary based on dataset):
- **2-Class**: 95-98% accuracy
- **8-Class**: 90-95% accuracy
- **34-Class**: 85-92% accuracy

### Training Time:
- **2-Class**: ~15-30 minutes
- **8-Class**: ~25-45 minutes
- **34-Class**: ~45-90 minutes

*Note: Training time depends on hardware specifications and dataset size*

## Dataset Requirements

### Format
- CSV file with features and target column
- Target column should be named 'label' or 'Label'
- All features should be numerical or categorical

### Expected Columns
- Multiple network traffic features
- Target column with attack labels
- Recommended minimum: 1000 samples per class

## Troubleshooting

### Common Issues

1. **Memory Error**
   ```python
   # Reduce dataset size or use sampling
   df_sample = df.sample(n=100000, random_state=42)
   ```

2. **Convergence Issues**
   ```python
   # Increase GOA iterations
   goa = AdvancedGazelleOptimizationAlgorithm(max_iterations=200)
   ```

3. **Class Imbalance**
   ```python
   # Adjust target samples in balancing
   df_balanced = self.balance_multiclass_advanced(df, target_samples=3000)
   ```

### Performance Tips

1. **For Large Datasets**: Use sampling for initial optimization
2. **For Small Datasets**: Increase cross-validation folds
3. **For Imbalanced Data**: Adjust SMOTE parameters
4. **For Speed**: Reduce GOA population size and iterations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this algorithm in your research, please cite:

```bibtex
@misc{enhanced_ciciot2023_ml,
  title={Enhanced Machine Learning Algorithm for CICIoT2023 Dataset with Gazelle Optimization},
  author={Your Name},
  year={2024},
  howpublished={GitHub Repository},
  url={https://github.com/yourusername/enhanced-ciciot2023-ml}
}
```

## Acknowledgments

- CICIoT2023 dataset creators
- Gazelle Optimization Algorithm researchers
- Open-source machine learning community
- Contributors to scikit-learn, XGBoost, and LightGBM

## Support

For questions, issues, or suggestions:
- Create an issue on GitHub
- Email: your.email@example.com
- Documentation: [Link to documentation]

## Changelog

### Version 1.0.0 (2024-01-01)
- Initial release
- Enhanced GOA implementation
- Hybrid ensemble model
- Multi-class representation support
- Advanced preprocessing pipeline

### Version 1.1.0 (2024-01-15)
- Added LightGBM integration
- Improved class balancing
- Enhanced feature engineering
- Bug fixes and performance improvements

---

**Note**: This algorithm is specifically designed for the CICIoT2023 dataset but can be adapted for other cybersecurity datasets with similar characteristics.
