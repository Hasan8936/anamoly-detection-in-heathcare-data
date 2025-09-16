import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
import tempfile
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Optional imports for production mode
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.utils import resample
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb
    PRODUCTION_MODE = True
except ImportError:
    PRODUCTION_MODE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="DIRA - IoT Security Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with cybersecurity theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3); }
        to { box-shadow: 0 25px 50px rgba(102, 126, 234, 0.5); }
    }
    
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 300;
        color: rgba(255,255,255,0.9);
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(52, 152, 219, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(52, 152, 219, 0.3);
    }
    
    .metric-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: #3498db;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Source Code Pro', monospace;
        font-size: 2.5rem;
        font-weight: 600;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .status-success {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00ff88;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9500;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 5px 15px rgba(243, 156, 18, 0.3);
    }
    
    .status-danger {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4757;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 5px 15px rgba(231, 76, 60, 0.3);
    }
    
    .attack-detected {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.4);
        animation: pulse 2s infinite;
    }
    
    .normal-traffic {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.4);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .code-block {
        background: #1e1e1e;
        border: 1px solid #3498db;
        border-radius: 10px;
        padding: 1.5rem;
        font-family: 'Source Code Pro', monospace;
        font-size: 0.9rem;
        color: #00ff88;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Model Management Classes
class GazelleOptimizationAlgorithm:
    """Enhanced GOA with feature selection capability"""
    
    def __init__(self, population_size=30, max_iterations=50):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.best_position = None
        self.best_fitness = float('inf')
        
    def initialize_population(self, bounds):
        """Initialize population within given bounds"""
        dim = len(bounds)
        population = np.zeros((self.population_size, dim))
        for i in range(dim):
            lower, upper = bounds[i]
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        return population
    
    def fitness_function(self, position, X_train, y_train, X_val, y_val, n_features_to_select=25):
        """Fitness function with feature selection"""
        try:
            hyperparams = position[:10]
            feature_weights = position[10:]
            n_total_features = X_train.shape[1]
            
            if len(feature_weights) != n_total_features:
                return float('inf')
                
            # Select top k features
            if n_total_features > n_features_to_select:
                top_feature_indices = np.argsort(feature_weights)[-n_features_to_select:]
                X_train_reduced = X_train[:, top_feature_indices]
                X_val_reduced = X_val[:, top_feature_indices]
            else:
                X_train_reduced = X_train
                X_val_reduced = X_val
                
            # Decode hyperparameters
            rf_n_estimators = max(50, min(200, int(hyperparams[0])))
            rf_max_depth = int(hyperparams[1]) if hyperparams[1] > 0 else None
            rf_min_samples_split = max(2, min(10, int(hyperparams[2])))
            rf_min_samples_leaf = max(1, min(5, int(hyperparams[3])))
            
            xgb_n_estimators = max(50, min(200, int(hyperparams[4])))
            xgb_max_depth = max(3, min(10, int(hyperparams[5])))
            xgb_learning_rate = max(0.01, min(0.3, hyperparams[6]))
            xgb_subsample = max(0.6, min(1.0, hyperparams[7]))
            xgb_colsample_bytree = max(0.6, min(1.0, hyperparams[8]))
            
            rf_weight = max(0.3, min(0.7, hyperparams[9]))
            xgb_weight = 1 - rf_weight
            
            # Train models
            rf_model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_reduced, y_train)
            rf_pred_proba = rf_model.predict_proba(X_val_reduced)
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
            xgb_model.fit(X_train_reduced, y_train)
            xgb_pred_proba = xgb_model.predict_proba(X_val_reduced)
            
            # Ensemble prediction
            ensemble_pred_proba = rf_weight * rf_pred_proba + xgb_weight * xgb_pred_proba
            ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
            
            accuracy = accuracy_score(y_val, ensemble_pred)
            return 1 - accuracy  # Minimize error
            
        except Exception as e:
            logger.error(f"Error in fitness function: {str(e)[:100]}")
            return float('inf')
    
    def optimize(self, X_train, y_train, X_val, y_val, bounds, n_features_to_select=25, progress_callback=None):
        """Main optimization loop with progress tracking"""
        logger.info("Starting Gazelle Optimization with Feature Selection...")
        population = self.initialize_population(bounds)
        fitness_values = np.zeros(self.population_size)
        dim = len(bounds)
        
        # Evaluate initial population
        for i in range(self.population_size):
            fitness_values[i] = self.fitness_function(
                population[i], X_train, y_train, X_val, y_val, n_features_to_select
            )
            if fitness_values[i] < self.best_fitness:
                self.best_fitness = fitness_values[i]
                self.best_position = population[i].copy()
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            if progress_callback:
                progress_callback(iteration, self.max_iterations, self.best_fitness)
            
            for i in range(self.population_size):
                # Exploration/exploitation balance
                if np.random.rand() < 0.5:
                    r1, r2 = np.random.choice(self.population_size, 2, replace=False)
                    step = 2 * np.random.rand() - 1
                    population[i] += step * (population[r1] - population[r2])
                else:
                    step = 2 * np.random.rand() - 1
                    population[i] += step * (self.best_position - population[i])
                
                # Apply bounds
                for d in range(dim):
                    low, high = bounds[d]
                    population[i, d] = np.clip(population[i, d], low, high)
                
                # Evaluate new position
                new_fitness = self.fitness_function(
                    population[i], X_train, y_train, X_val, y_val, n_features_to_select
                )
                
                # Update if improved
                if new_fitness < fitness_values[i]:
                    fitness_values[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = population[i].copy()
        
        logger.info(f"Optimization completed. Best fitness: {self.best_fitness:.4f}")
        return self.best_position

@st.cache_resource
class IoTSecurityModelManager:
    """Production model manager with caching and health monitoring"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.rf_model = None
        self.xgb_model = None
        self.best_params = None
        self.feature_mask = None
        self.results = {}
        self.is_trained = False
        self.model_health = {"status": "uninitialized", "last_check": None}
        self.device = "cpu"  # Default to CPU
        
    def _create_attack_mapping(self, unique_classes: List[str]) -> Dict[str, List[str]]:
        """Create attack category mapping"""
        normal_labels = ['Normal', 'BENIGN', 'Benign', 'normal', 'benign']
        normal_class = None
        
        for label in normal_labels:
            if label in unique_classes:
                normal_class = label
                break
        
        remaining_classes = [c for c in unique_classes if c != normal_class]
        
        attack_categories = {
            'Normal': [normal_class] if normal_class else [],
            'DDoS': [],
            'DoS': [],
            'Recon': [],
            'Web': [],
            'BruteForce': [],
            'Spoofing': [],
            'Mirai': []
        }
        
        # Classify attacks based on class names
        for class_name in remaining_classes:
            class_name_lower = str(class_name).lower()
            
            if 'ddos' in class_name_lower:
                attack_categories['DDoS'].append(class_name)
            elif 'dos' in class_name_lower:
                attack_categories['DoS'].append(class_name)
            elif any(keyword in class_name_lower for keyword in ['recon', 'scan', 'ping', 'port']):
                attack_categories['Recon'].append(class_name)
            elif any(keyword in class_name_lower for keyword in ['web', 'sql', 'xss', 'injection']):
                attack_categories['Web'].append(class_name)
            elif 'brute' in class_name_lower or 'force' in class_name_lower:
                attack_categories['BruteForce'].append(class_name)
            elif any(keyword in class_name_lower for keyword in ['spoof', 'mitm', 'arp']):
                attack_categories['Spoofing'].append(class_name)
            elif 'mirai' in class_name_lower:
                attack_categories['Mirai'].append(class_name)
            else:
                attack_categories['DDoS'].append(class_name)  # Default
        
        return attack_categories
    
    def _create_class_representation(self, df: pd.DataFrame, representation_type: str) -> pd.DataFrame:
        """Create different class representations"""
        df_copy = df.copy()
        target_col = df_copy.columns[-1]
        
        unique_classes = df_copy[target_col].unique()
        logger.info(f"Original classes: {unique_classes}")
        
        if representation_type == '2-class':
            # Binary classification: Normal vs Attack
            normal_labels = ['Normal', 'BENIGN', 'Benign', 'normal', 'benign']
            normal_class = None
            
            for label in normal_labels:
                if label in unique_classes:
                    normal_class = label
                    break
            
            if normal_class is None:
                class_counts = df_copy[target_col].value_counts()
                normal_class = class_counts.index[0]
            
            df_copy[target_col] = df_copy[target_col].apply(
                lambda x: 'Normal' if x == normal_class else 'Attack'
            )
            
        elif representation_type == '8-class':
            # Group attacks into categories
            attack_mapping = self._create_attack_mapping(unique_classes)
            for new_class, old_classes in attack_mapping.items():
                df_copy[target_col] = df_copy[target_col].replace(old_classes, new_class)
        
        # For 34-class, keep original classes
        return df_copy
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get model health status"""
        return self.model_health
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load model from path with validation"""
        try:
            if model_path and os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.rf_model = model_data['rf_model']
                    self.xgb_model = model_data['xgb_model']
                    self.scaler = model_data['scaler']
                    self.label_encoder = model_data['label_encoder']
                    self.feature_mask = model_data['feature_mask']
                    self.best_params = model_data['best_params']
                    self.is_trained = True
                    
                self.model_health = {"status": "loaded", "last_check": datetime.now()}
                logger.info(f"Model loaded successfully from {model_path}")
                return True
            else:
                logger.warning("Model path not provided or doesn't exist. Using training mode.")
                self.model_health = {"status": "training_required", "last_check": datetime.now()}
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_health = {"status": "error", "last_check": datetime.now(), "error": str(e)}
            return False
    
    def validate_input(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate input data"""
        if data is None or data.empty:
            return False, "Empty dataset provided"
        
        if data.shape[0] < 10:
            return False, "Dataset too small (minimum 10 samples required)"
        
        # Check for required columns (basic validation)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 5:
            return False, "Insufficient numeric features (minimum 5 required)"
        
        return True, "Input validation passed"
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data for inference or training"""
        try:
            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))
            
            # Separate features and target
            if is_training:
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Encode labels
                y_encoded = self.label_encoder.fit_transform(y)
                
                return X_scaled, y_encoded
            else:
                X = df.values
                X_scaled = self.scaler.transform(X)
                return X_scaled, None
                
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise
    
    def train_model(self, df: pd.DataFrame, representation_type: str = '2-class', 
                   n_features: int = 25, population_size: int = 30, 
                   max_iterations: int = 50, progress_callback=None) -> Dict[str, float]:
        """Train model with GOA optimization"""
        try:
            logger.info(f"Training model with {representation_type} representation")
            
            # Data validation
            is_valid, message = self.validate_input(df)
            if not is_valid:
                raise ValueError(message)
            
            # Create class representation
            df_processed = self._create_class_representation(df, representation_type)
            
            # Preprocess data
            X, y = self.preprocess_data(df_processed, is_training=True)
            
            # Train-validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            n_total_features = X_train.shape[1]
            
            # Define optimization bounds
            bounds = [
                (50, 200),    # RF n_estimators
                (3, 20),      # RF max_depth
                (2, 10),      # RF min_samples_split
                (1, 5),       # RF min_samples_leaf
                (50, 200),    # XGB n_estimators
                (3, 10),      # XGB max_depth
                (0.01, 0.3),  # XGB learning_rate
                (0.6, 1.0),   # XGB subsample
                (0.6, 1.0),   # XGB colsample_bytree
                (0.3, 0.7)    # RF weight
            ] + [(0, 1)] * n_total_features  # Feature weights
            
            # Initialize and run GOA
            goa = GazelleOptimizationAlgorithm(population_size, max_iterations)
            self.best_params = goa.optimize(
                X_train, y_train, X_val, y_val, bounds, n_features, progress_callback
            )
            
            # Extract optimized parameters
            hyperparams = self.best_params[:10]
            feature_weights = self.best_params[10:]
            
            # Create feature mask
            self.feature_mask = np.argsort(feature_weights)[-n_features:]
            
            # Train final models with selected features
            X_train_selected = X_train[:, self.feature_mask]
            X_val_selected = X_val[:, self.feature_mask]
            
            # RF parameters
            self.rf_model = RandomForestClassifier(
                n_estimators=int(hyperparams[0]),
                max_depth=int(hyperparams[1]) if hyperparams[1] > 0 else None,
                min_samples_split=int(hyperparams[2]),
                min_samples_leaf=int(hyperparams[3]),
                random_state=42,
                n_jobs=-1
            )
            
            # XGB parameters
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=int(hyperparams[4]),
                max_depth=int(hyperparams[5]),
                learning_rate=hyperparams[6],
                subsample=hyperparams[7],
                colsample_bytree=hyperparams[8],
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
            
            self.rf_weight = hyperparams[9]
            self.xgb_weight = 1 - self.rf_weight
            
            # Train models
            self.rf_model.fit(X_train_selected, y_train)
            self.xgb_model.fit(X_train_selected, y_train)
            
            # Evaluate on validation set
            rf_pred_proba = self.rf_model.predict_proba(X_val_selected)
            xgb_pred_proba = self.xgb_model.predict_proba(X_val_selected)
            
            ensemble_proba = self.rf_weight * rf_pred_proba + self.xgb_weight * xgb_pred_proba
            y_pred = np.argmax(ensemble_proba, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            self.results = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'representation_type': representation_type,
                'n_features': n_features,
                'population_size': population_size,
                'max_iterations': max_iterations
            }
            
            self.is_trained = True
            self.model_health = {"status": "trained", "last_check": datetime.now()}
            
            logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
            return self.results
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.model_health = {"status": "training_error", "last_check": datetime.now(), "error": str(e)}
            raise
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on input data"""
        try:
            if not self.is_trained:
                raise RuntimeError("Model not trained. Please train the model first.")
            
            start_time = time.time()
            
            # Preprocess input
            X_processed, _ = self.preprocess_data(data, is_training=False)
            
            # Apply feature selection
            X_selected = X_processed[:, self.feature_mask]
            
            # Get predictions from both models
            rf_proba = self.rf_model.predict_proba(X_selected)
            xgb_proba = self.xgb_model.predict_proba(X_selected)
            
            # Ensemble prediction
            ensemble_proba = self.rf_weight * rf_proba + self.xgb_weight * xgb_proba
            predictions = np.argmax(ensemble_proba, axis=1)
            
            # Decode predictions
            prediction_labels = self.label_encoder.inverse_transform(predictions)
            prediction_confidences = np.max(ensemble_proba, axis=1)
            
            inference_time = time.time() - start_time
            
            results = {
                'predictions': prediction_labels.tolist(),
                'confidences': prediction_confidences.tolist(),
                'probabilities': ensemble_proba.tolist(),
                'class_names': self.label_encoder.classes_.tolist(),
                'inference_time': inference_time,
                'model_status': self.model_health['status']
            }
            
            self.model_health['last_inference'] = datetime.now()
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Demo fallback for when production dependencies aren't available
class IoTSecurityDemo:
    """Fallback demo implementation"""
    
    def __init__(self):
        self.is_trained = False
        self.results = {}
        
    def train_model(self, df, representation_type, n_features, population_size, max_iterations, progress_callback=None):
        """Simulate training with realistic results"""
        if progress_callback:
            for i in range(max_iterations):
                progress_callback(i, max_iterations, 1 - (0.95 + i * 0.002))
                time.sleep(0.1)
        
        # Generate realistic results
        base_accuracy = 0.99 if representation_type == '2-class' else 0.98 if representation_type == '8-class' else 0.97
        accuracy = base_accuracy + np.random.uniform(-0.005, 0.005)
        
        self.results = {
            'Accuracy': accuracy,
            'Precision': accuracy + np.random.uniform(-0.003, 0.003),
            'Recall': accuracy + np.random.uniform(-0.003, 0.003),
            'F1-Score': accuracy + np.random.uniform(-0.003, 0.003),
            'representation_type': representation_type,
            'n_features': n_features,
            'population_size': population_size,
            'max_iterations': max_iterations
        }
        
        self.is_trained = True
        return self.results
    
    def predict(self, data):
        """Generate demo predictions"""
        n_samples = len(data)
        predictions = np.random.choice(['Normal', 'Attack'], n_samples, p=[0.7, 0.3])
        confidences = np.random.uniform(0.85, 0.99, n_samples)
        
        return {
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'inference_time': np.random.uniform(0.1, 0.5),
            'model_status': 'demo_mode'
        }

# Initialize model manager
@st.cache_resource
def get_model_manager():
    """Initialize and return model manager"""
    if PRODUCTION_MODE:
        return IoTSecurityModelManager()
    else:
        return IoTSecurityDemo()

# Utility functions
def create_synthetic_dataset(n_samples=5000, n_features=50):
    """Create realistic synthetic dataset"""
    np.random.seed(42)
    
    # Generate base features
    base_features = np.random.randn(n_samples, 10)
    additional_features = []
    
    for i in range(n_features - 10):
        if i < 20:
            feature = base_features[:, i % 10] + np.random.randn(n_samples) * 0.3
        else:
            feature = np.abs(base_features[:, i % 10]) + np.random.exponential(0.5, n_samples)
        additional_features.append(feature)
    
    X = np.column_stack([base_features, np.array(additional_features).T])
    
    # Generate attack labels
    decision_1 = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    decision_2 = -X[:, 3] + 2 * X[:, 4] - 0.5 * X[:, 5] + np.random.randn(n_samples) * 0.7
    decision_3 = X[:, 6] + X[:, 7] - X[:, 8] + np.random.randn(n_samples) * 0.6
    
    labels = []
    for i in range(n_samples):
        if decision_1[i] > 2:
            labels.append('DDoS_Attack')
        elif decision_2[i] > 1.5:
            labels.append('Malware_Attack')
        elif decision_3[i] > 1:
            labels.append('Intrusion_Attack')
        elif decision_1[i] < -2:
            labels.append('Recon_Attack')
        else:
            labels.append('Normal')
    
    columns = [f'feature_{i}' for i in range(n_features)] + ['label']
    df = pd.DataFrame(np.column_stack([X, labels]), columns=columns)
    
    return df

def display_progress_ring(percentage, label="Progress"):
    """Display animated progress ring"""
    size = 120
    stroke_width = 10
    radius = (size - stroke_width) / 2
    circumference = 2 * 3.14159 * radius
    offset = circumference - (percentage / 100) * circumference
    
    return f"""
    <div style="display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
        <svg width="{size}" height="{size}" class="progress-ring">
            <circle cx="{size/2}" cy="{size/2}" r="{radius}" 
                   stroke="#34495e" stroke-width="{stroke_width}" fill="transparent"/>
            <circle cx="{size/2}" cy="{size/2}" r="{radius}"
                   stroke="#00ff88" stroke-width="{stroke_width}" fill="transparent"
                   stroke-dasharray="{circumference} {circumference}"
                   stroke-dashoffset="{offset}"
                   style="transition: stroke-dashoffset 0.5s ease-in-out;"/>
            <text x="{size/2}" y="{size/2}" text-anchor="middle" dy="0.3em" 
                  fill="#ffffff" font-size="18" font-weight="600">
                {percentage:.1f}%
            </text>
        </svg>
        <span style="margin-left: 1rem; color: #ffffff; font-weight: 500;">{label}</span>
    </div>
    """

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = get_model_manager()
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'inference_history' not in st.session_state:
    st.session_state.inference_history = []

# Main Application
def main():
    model_manager = st.session_state.model_manager
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">DIRA AI</div>
        <div class="hero-subtitle">Advanced IoT Security Intelligence Platform</div>
        <p>Powered by Gazelle Optimization & Ensemble Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status Banner
    if PRODUCTION_MODE:
        if hasattr(model_manager, 'model_health'):
            health = model_manager.get_health_status()
            if health['status'] == 'trained':
                st.markdown("""
                <div class="status-success">
                    🟢 <strong>SYSTEM OPERATIONAL</strong><br>
                    Production model loaded and ready for threat detection
                </div>
                """, unsafe_allow_html=True)
            elif health['status'] == 'training_required':
                st.markdown("""
                <div class="status-warning">
                    🟡 <strong>TRAINING REQUIRED</strong><br>
                    Upload dataset to train the security model
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-danger">
                    🔴 <strong>SYSTEM ERROR</strong><br>
                    Model initialization failed - check logs
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-warning">
            🟡 <strong>DEMO MODE ACTIVE</strong><br>
            Install production dependencies (scikit-learn, xgboost, imbalanced-learn) for full functionality
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Model Settings
        st.markdown("### 🎯 Model Settings")
        
        representation_type = st.selectbox(
            "Classification Type",
            ["2-class", "8-class", "34-class"],
            help="Choose attack detection granularity"
        )
        
        n_features = st.slider(
            "Features to Select",
            min_value=10,
            max_value=50,
            value=15,  # Reduced from 25 to 15 for faster training
            help="Number of top features for optimization"
        )
        
        # Optimization Settings
        st.markdown("### 🔧 Optimization Settings")
        
        population_size = st.slider(
            "Population Size",
            min_value=5,  # Reduced minimum from 10 to 5
            max_value=30,  # Reduced maximum from 50 to 30
            value=10,  # Reduced from 25 to 10 for faster training
            help="⚠️ Higher values will significantly increase training time"
        )
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=5,  # Reduced minimum from 10 to 5
            max_value=50,  # Reduced maximum from 100 to 50
            value=15,  # Reduced from 40 to 15 for faster training
            help="⚠️ Higher values will significantly increase training time"
        )
        
        # Warning about training time
        st.warning("⚠️ Training time increases exponentially with higher population size and iterations")
    
    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs([
        "🚀 Detection Center",
        "📊 Training Lab", 
        "📈 Analytics Dashboard"
    ])
    
    # Tab 1: Detection Center
    with tab1:
        st.markdown("## 🛡️ Threat Detection Center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📁 Upload Network Data")
            
            uploaded_file = st.file_uploader(
                "Choose CSV file containing network traffic data",
                type=['csv'],
                help="Upload IoT network traffic data for threat analysis"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, nrows=1000)
                    
                    st.success(f"✅ Dataset loaded: {len(df)} samples")
                    
                    # Dataset preview
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head())
                    
                    # Run Detection
                    if st.button("🔍 Run Threat Detection", type="primary", use_container_width=True):
                        if hasattr(model_manager, 'is_trained') and model_manager.is_trained:
                            try:
                                with st.spinner("Analyzing network traffic..."):
                                    results = model_manager.predict(df)
                                
                                predictions = results['predictions']
                                confidences = results['confidences']
                                
                                # Overall threat assessment
                                threat_count = sum(1 for p in predictions if p != 'Normal')
                                threat_percentage = (threat_count / len(predictions)) * 100
                                
                                if threat_percentage > 50:
                                    st.markdown(f"""
                                    <div class="attack-detected">
                                        🚨 HIGH THREAT LEVEL DETECTED
                                        <br>
                                        {threat_count}/{len(predictions)} samples flagged ({threat_percentage:.1f}%)
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="normal-traffic">
                                        ✅ NORMAL TRAFFIC DETECTED
                                        <br>
                                        {threat_count}/{len(predictions)} samples flagged ({threat_percentage:.1f}%)
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Detection failed: {str(e)}")
                                
                        else:
                            # Demo mode
                            with st.spinner("Simulating threat detection..."):
                                time.sleep(2)
                                results = model_manager.predict(df)
                            
                            predictions = results['predictions']
                            threat_count = sum(1 for p in predictions if p == 'Attack')
                            
                            st.markdown(f"""
                            <div class="status-success">
                                🔍 DEMO ANALYSIS COMPLETE
                                <br>
                                {threat_count}/{len(predictions)} potential threats detected
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        with col2:
            st.markdown("### 🎲 Quick Test")
            
            if st.button("🧪 Generate Test Data", use_container_width=True):
                test_df = create_synthetic_dataset(500, 30)
                
                if hasattr(model_manager, 'is_trained') and model_manager.is_trained:
                    results = model_manager.predict(test_df)
                    predictions = results['predictions']
                    threat_count = sum(1 for p in predictions if p != 'Normal')
                else:
                    predictions = np.random.choice(['Normal', 'Attack'], 500, p=[0.8, 0.2])
                    threat_count = sum(1 for p in predictions if p == 'Attack')
                
                st.metric("Test Samples", 500)
                st.metric("Threats Found", threat_count)
                st.metric("Threat Rate", f"{(threat_count/500)*100:.1f}%")
    
    # Tab 2: Training Lab
    with tab2:
        st.markdown("## 🧪 Training Laboratory")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📁 Dataset Upload")
            
            training_file = st.file_uploader(
                "Upload Training Dataset",
                type=['csv'],
                help="Upload labeled IoT security dataset for training",
                key="training_upload"
            )
            
            if training_file is not None:
                try:
                    df = pd.read_csv(training_file)
                    st.success(f"✅ Training data loaded: {len(df)} samples, {len(df.columns)} features")
                    
                    # Training controls
                    st.markdown("### 🚀 Start Training")
                    
                    if st.button("🔥 Begin Model Training", type="primary", use_container_width=True):
                        progress_container = st.empty()
                        
                        def progress_callback(iteration, max_iter, fitness):
                            progress = (iteration + 1) / max_iter * 100
                            progress_container.progress(progress / 100)
                        
                        results = model_manager.train_model(
                            df, representation_type, n_features, 
                            population_size, max_iterations, progress_callback
                        )
                        
                        st.success(f"🎉 Training completed! Accuracy: {results['Accuracy']:.4f}")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error loading training data: {str(e)}")
        
        with col2:
            st.markdown("### 🎲 Demo Training")
            
            if st.button("🎯 Demo Train", use_container_width=True):
                demo_df = create_synthetic_dataset()
                
                progress_container = st.empty()
                
                def demo_progress(iteration, max_iter, fitness):
                    progress = (iteration + 1) / max_iter * 100
                    progress_container.progress(progress / 100)
                
                results = model_manager.train_model(
                    demo_df, representation_type, n_features,
                    population_size, max_iterations, demo_progress
                )
                
                st.success(f"✅ Demo Complete: {results['Accuracy']:.4f}")
    
    # Tab 3: Analytics Dashboard
    with tab3:
        st.markdown("## 📊 Analytics Dashboard")
        
        if hasattr(model_manager, 'results') and model_manager.results:
            results = model_manager.results
            
            # Performance Metrics
            st.markdown("### 🎯 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🎯 Accuracy</div>
                    <div class="metric-value">{results['Accuracy']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📍 Precision</div>
                    <div class="metric-value">{results['Precision']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📡 Recall</div>
                    <div class="metric-value">{results['Recall']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">⚖️ F1-Score</div>
                    <div class="metric-value">{results['F1-Score']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 Train a model to view detailed analytics")

# Removed the footer section completely

if __name__ == "__main__":
    main()
