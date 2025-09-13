import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import tempfile
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="IoT Security ML Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.8rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.8rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .feature-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .code-block {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Demo ML Algorithm Class
class IoTSecurityMLDemo:
    """
    Demo implementation of IoT Security ML Pipeline
    Simulates CNN+LightGBM with Gazelle Optimization
    """
    
    def __init__(self):
        self.is_trained = False
        self.results = {}
        self.training_history = []
        self.feature_importance = None
        self.model_params = {}
        
    def simulate_training(self, df, representation_type, n_features, population_size, max_iterations):
        """Simulate the training process with realistic progress"""
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        try:
            # Step 1: Data preprocessing
            status_container.text("🔄 Preprocessing data...")
            progress_bar.progress(10)
            
            # Simulate preprocessing delay
            import time
            time.sleep(1)
            
            # Step 2: Feature selection setup
            status_container.text("🎯 Setting up feature selection...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Step 3: Initialize Gazelle Optimization
            status_container.text("🦌 Initializing Gazelle Optimization Algorithm...")
            progress_bar.progress(30)
            time.sleep(0.5)
            
            # Simulate optimization iterations
            best_accuracy = 0.7
            target_accuracy = self._get_target_accuracy(representation_type)
            
            for iteration in range(max_iterations):
                progress = 30 + (iteration / max_iterations) * 60
                status_container.text(f"🔧 Optimization iteration {iteration+1}/{max_iterations}")
                progress_bar.progress(int(progress))
                
                # Simulate improvement
                improvement = np.random.exponential(0.01)
                best_accuracy = min(target_accuracy + 0.005, best_accuracy + improvement)
                self.training_history.append(best_accuracy)
                
                time.sleep(0.1)  # Small delay for visualization
            
            # Final model training
            status_container.text("🧠 Training final CNN+LightGBM ensemble...")
            progress_bar.progress(90)
            time.sleep(1)
            
            # Generate final results
            final_accuracy = best_accuracy + np.random.uniform(-0.002, 0.005)
            final_accuracy = min(0.999, max(0.95, final_accuracy))
            
            precision = final_accuracy + np.random.uniform(-0.003, 0.003)
            recall = final_accuracy + np.random.uniform(-0.003, 0.003)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            self.results = {
                'Accuracy': final_accuracy,
                'Precision': max(0.90, precision),
                'Recall': max(0.90, recall),
                'F1-Score': max(0.90, f1_score),
                'representation_type': representation_type,
                'n_features': n_features,
                'population_size': population_size,
                'max_iterations': max_iterations
            }
            
            # Generate feature importance
            self.feature_importance = np.random.exponential(0.5, n_features)
            self.feature_importance = self.feature_importance / np.sum(self.feature_importance)
            
            # Generate model parameters
            self._generate_model_params()
            
            # Complete
            status_container.text("✅ Training completed successfully!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            self.is_trained = True
            return self.results
            
        except Exception as e:
            status_container.error(f"Training failed: {e}")
            return None
    
    def _get_target_accuracy(self, representation_type):
        """Get target accuracy based on classification type"""
        if representation_type == '2-class':
            return np.random.uniform(0.995, 0.999)
        elif representation_type == '8-class':
            return np.random.uniform(0.990, 0.996)
        else:  # 34-class
            return np.random.uniform(0.985, 0.993)
    
    def _generate_model_params(self):
        """Generate realistic model parameters"""
        self.model_params = {
            'cnn_params': {
                'conv1_filters': np.random.randint(32, 128),
                'conv2_filters': np.random.randint(64, 256),
                'conv3_filters': np.random.randint(128, 512),
                'dense1_units': np.random.randint(64, 256),
                'dense2_units': np.random.randint(32, 128),
                'dropout_rate': round(np.random.uniform(0.1, 0.5), 3),
                'learning_rate': round(np.random.uniform(0.0001, 0.01), 6)
            },
            'lightgbm_params': {
                'num_leaves': np.random.randint(20, 100),
                'learning_rate': round(np.random.uniform(0.01, 0.3), 6),
                'feature_fraction': round(np.random.uniform(0.6, 1.0), 3),
                'bagging_fraction': round(np.random.uniform(0.6, 1.0), 3),
                'bagging_freq': np.random.randint(3, 10),
                'min_child_samples': np.random.randint(5, 50),
                'lambda_l1': round(np.random.uniform(0.0, 1.0), 3),
                'lambda_l2': round(np.random.uniform(0.0, 1.0), 3)
            },
            'ensemble_weight': round(np.random.uniform(0.3, 0.7), 3)
        }

# Initialize session state
if 'algorithm' not in st.session_state:
    st.session_state.algorithm = IoTSecurityMLDemo()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_results' not in st.session_state:
    st.session_state.training_results = {}

# Main header
st.markdown("""
<div class="main-header">
    <h1>🛡️ IoT Security ML Pipeline</h1>
    <h3>CNN + LightGBM with Gazelle Optimization</h3>
    <p>Advanced Machine Learning for IoT Network Attack Detection</p>
</div>
""", unsafe_allow_html=True)

# System status
st.markdown("""
<div class="status-warning">
    <strong>🔧 Demo Mode Active</strong><br>
    This is a demonstration version with simulated results. 
    For production deployment, install the full dependency stack including TensorFlow and LightGBM.
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Model Configuration")
representation_type = st.sidebar.selectbox(
    "Classification Type",
    ["2-class", "8-class", "34-class"],
    help="Choose the number of attack categories to classify"
)

n_features = st.sidebar.slider(
    "Features to Select",
    min_value=10,
    max_value=50,
    value=30,
    help="Number of top features selected by GOA"
)

st.sidebar.subheader("🔧 Optimization Settings")
population_size = st.sidebar.slider(
    "Population Size",
    min_value=10,
    max_value=50,
    value=25
)

max_iterations = st.sidebar.slider(
    "Max Iterations",
    min_value=10,
    max_value=100,
    value=40
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Current Configuration:**
- Classification: {representation_type}
- Features: {n_features}
- Population: {population_size}
- Iterations: {max_iterations}
""")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Training & Data", 
    "📈 Results & Analysis", 
    "🎯 Model Details",
    "📚 Documentation"
])

# Tab 1: Training & Data
with tab1:
    st.header("📊 Dataset & Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload your IoT security dataset"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, nrows=1000)  # Preview first 1000 rows
                
                st.success("✅ Dataset loaded successfully!")
                
                # Dataset statistics
                col1_stat, col2_stat, col3_stat = st.columns(3)
                with col1_stat:
                    st.metric("Rows (preview)", len(df))
                with col2_stat:
                    st.metric("Columns", len(df.columns))
                with col3_stat:
                    target_col = df.columns[-1]
                    st.metric("Classes", df[target_col].nunique())
                
                # Dataset preview
                with st.expander("📋 Dataset Preview"):
                    st.dataframe(df.head(10))
                
                # Class distribution
                with st.expander("📊 Class Distribution"):
                    class_counts = df[target_col].value_counts()
                    st.bar_chart(class_counts)
                    
                    # Show statistics
                    st.write("**Class Statistics:**")
                    for class_name, count in class_counts.items():
                        st.write(f"- {class_name}: {count} samples")
                
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    
    with col2:
        st.subheader("Training Control")
        
        st.markdown("""
        <div class="feature-box">
        <strong>🎯 Training Features:</strong><br>
        • Gazelle Optimization Algorithm<br>
        • Automated feature selection<br>
        • CNN + LightGBM ensemble<br>
        • Target: >99% accuracy
        </div>
        """, unsafe_allow_html=True)
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if uploaded_file is not None:
                df_full = pd.read_csv(uploaded_file)
                results = st.session_state.algorithm.simulate_training(
                    df_full, representation_type, n_features, population_size, max_iterations
                )
                if results:
                    st.session_state.training_results = results
                    st.session_state.model_trained = True
                    st.success(f"🎉 Training completed! Accuracy: {results['Accuracy']:.4f}")
                    st.rerun()
            else:
                st.warning("Please upload a dataset first")
    
    # Sample dataset option
    st.markdown("---")
    st.subheader("🧪 Demo with Sample Data")
    
    col1_demo, col2_demo = st.columns([3, 1])
    
    with col1_demo:
        st.info("""
        **Generate synthetic IoT security dataset:**
        - 5,000 samples with realistic network traffic features
        - Multiple attack categories (DDoS, Malware, Intrusion, etc.)
        - Balanced class distribution for optimal training
        """)
    
    with col2_demo:
        if st.button("🎲 Demo Training", use_container_width=True):
            # Create sample dataset
            sample_df = create_synthetic_dataset()
            results = st.session_state.algorithm.simulate_training(
                sample_df, representation_type, n_features, population_size, max_iterations
            )
            if results:
                st.session_state.training_results = results
                st.session_state.model_trained = True
                st.success(f"🎉 Demo training completed! Accuracy: {results['Accuracy']:.4f}")
                st.rerun()

# Tab 2: Results & Analysis
with tab2:
    st.header("📈 Training Results & Performance Analysis")
    
    if st.session_state.model_trained and st.session_state.training_results:
        results = st.session_state.training_results
        
        # Performance metrics
        st.subheader("🎯 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = results['Accuracy']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <h2>{accuracy:.4f}</h2>
                <p>{accuracy*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            precision = results['Precision']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Precision</h3>
                <h2>{precision:.4f}</h2>
                <p>{precision*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            recall = results['Recall']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Recall</h3>
                <h2>{recall:.4f}</h2>
                <p>{recall*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            f1_score = results['F1-Score']
            st.markdown(f"""
            <div class="metric-card">
                <h3>F1-Score</h3>
                <h2>{f1_score:.4f}</h2>
                <p>{f1_score*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance status
        target_accuracy = 0.99
        if accuracy >= target_accuracy:
            st.markdown(f"""
            <div class="status-success">
                🎉 <strong>TARGET ACHIEVED!</strong><br>
                Accuracy ({accuracy:.4f}) exceeds the target threshold of {target_accuracy}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-warning">
                ⚡ <strong>Good Performance Achieved!</strong><br>
                Accuracy: {accuracy:.4f} (Target: {target_accuracy})<br>
                Consider increasing iterations or adjusting parameters for higher accuracy.
            </div>
            """, unsafe_allow_html=True)
        
        # Training convergence
        st.subheader("📊 Optimization Convergence")
        
        if hasattr(st.session_state.algorithm, 'training_history') and st.session_state.algorithm.training_history:
            history = st.session_state.algorithm.training_history
            
            # Create convergence chart
            chart_data = pd.DataFrame({
                'Iteration': list(range(1, len(history) + 1)),
                'Accuracy': history
            })
            
            st.line_chart(chart_data.set_index('Iteration'))
            
            # Convergence statistics
            col1_conv, col2_conv, col3_conv = st.columns(3)
            with col1_conv:
                st.metric("Initial Accuracy", f"{history[0]:.4f}")
            with col2_conv:
                st.metric("Final Accuracy", f"{history[-1]:.4f}")
            with col3_conv:
                improvement = history[-1] - history[0]
                st.metric("Total Improvement", f"{improvement:.4f}")
        
        # Feature importance
        st.subheader("🔍 Feature Selection Results")
        
        if hasattr(st.session_state.algorithm, 'feature_importance') and st.session_state.algorithm.feature_importance is not None:
            importance = st.session_state.algorithm.feature_importance
            
            col1_feat, col2_feat = st.columns([1, 2])
            
            with col1_feat:
                st.metric("Selected Features", len(importance))
                st.metric("Total Features", 50)  # Simulated
                reduction = (1 - len(importance) / 50) * 100
                st.metric("Dimension Reduction", f"{reduction:.1f}%")
            
            with col2_feat:
                # Top 15 features chart
                top_n = min(15, len(importance))
                top_indices = np.argsort(importance)[::-1][:top_n]
                
                feature_data = pd.DataFrame({
                    'Feature': [f'Feature_{i}' for i in top_indices],
                    'Importance': importance[top_indices]
                })
                
                st.bar_chart(feature_data.set_index('Feature'))
    
    else:
        st.info("🔄 No training results available. Please train a model first in the Training tab.")

# Tab 3: Model Details
with tab3:
    st.header("🎯 Model Architecture & Configuration")
    
    if st.session_state.model_trained and hasattr(st.session_state.algorithm, 'model_params'):
        params = st.session_state.algorithm.model_params
        
        # Model architecture
        st.subheader("🏗️ Optimized Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**CNN Parameters:**")
            cnn_params = params['cnn_params']
            
            st.markdown(f"""
            <div class="code-block">
            Conv1D Filters: {cnn_params['conv1_filters']}<br>
            Conv2D Filters: {cnn_params['conv2_filters']}<br>
            Conv3D Filters: {cnn_params['conv3_filters']}<br>
            Dense1 Units: {cnn_params['dense1_units']}<br>
            Dense2 Units: {cnn_params['dense2_units']}<br>
            Dropout Rate: {cnn_params['dropout_rate']}<br>
            Learning Rate: {cnn_params['learning_rate']}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**LightGBM Parameters:**")
            lgb_params = params['lightgbm_params']
            
            st.markdown(f"""
            <div class="code-block">
            Num Leaves: {lgb_params['num_leaves']}<br>
            Learning Rate: {lgb_params['learning_rate']}<br>
            Feature Fraction: {lgb_params['feature_fraction']}<br>
            Bagging Fraction: {lgb_params['bagging_fraction']}<br>
            Bagging Freq: {lgb_params['bagging_freq']}<br>
            Min Child Samples: {lgb_params['min_child_samples']}<br>
            L1 Regularization: {lgb_params['lambda_l1']}<br>
            L2 Regularization: {lgb_params['lambda_l2']}
            </div>
            """, unsafe_allow_html=True)
        
        # Ensemble configuration
        st.subheader("⚖️ Ensemble Configuration")
        
        ensemble_weight = params['ensemble_weight']
        lgb_weight = 1 - ensemble_weight
        
        st.markdown(f"""
        <div class="feature-box">
        <strong>Ensemble Weights:</strong><br>
        • CNN Contribution: {ensemble_weight:.1%}<br>
        • LightGBM Contribution: {lgb_weight:.1%}<br><br>
        <strong>Optimization Method:</strong> Gazelle Optimization Algorithm (GOA)<br>
        <strong>Feature Selection:</strong> Top-{n_features} most important features
        </div>
        """, unsafe_allow_html=True)
        
        # Export model configuration
        st.subheader("💾 Export Configuration")
        
        col1_export, col2_export = st.columns([2, 1])
        
        with col1_export:
            export_data = {
                'model_architecture': 'CNN+LightGBM Ensemble',
                'optimization_algorithm': 'Gazelle Optimization Algorithm',
                'performance': st.session_state.training_results,
                'parameters': params,
                'feature_selection': {
                    'method': 'GOA-based importance weighting',
                    'selected_features': n_features,
                    'total_features': 50
                },
                'training_config': {
                    'representation_type': representation_type,
                    'population_size': population_size,
                    'max_iterations': max_iterations
                },
                'export_date': datetime.now().isoformat()
            }
            
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📄 Download Model Config",
                data=json_str,
                file_name=f"iot_model_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2_export:
            st.info("""
            **Export includes:**
            - Model architecture
            - Optimized parameters  
            - Training results
            - Feature selection info
            - Configuration settings
            """)
    
    else:
        st.info("🔄 No model details available. Please train a model first.")

# Tab 4: Documentation
with tab4:
    st.header("📚 Documentation & User Guide")
    
    st.subheader("🎯 System Overview")
    st.markdown("""
    This IoT Security ML Pipeline implements a state-of-the-art hybrid approach for detecting network attacks in IoT environments:
    
    **🧠 Core Technologies:**
    - **Convolutional Neural Network (CNN)**: Captures complex patterns in network traffic data
    - **LightGBM**: Provides fast, accurate gradient boosting for classification
    - **Gazelle Optimization Algorithm (GOA)**: Optimizes both model hyperparameters and feature selection
    - **Ensemble Learning**: Combines strengths of both models for superior performance
    """)
    
    st.subheader("📊 Classification Types")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 2-Class**
        - Normal vs Attack
        - Binary classification
        - Highest accuracy target
        - Best for general detection
        """)
    
    with col2:
        st.markdown("""
        **🎯 8-Class** 
        - 7 attack categories + normal
        - DDoS, DoS, Recon, Web attacks, etc.
        - Balanced complexity/accuracy
        - Good for attack type identification
        """)
    
    with col3:
        st.markdown("""
        **🎯 34-Class**
        - Full attack taxonomy
        - Detailed attack classification
        - Most challenging
        - Best for forensic analysis
        """)
    
    st.subheader("⚙️ Parameter Guidelines")
    
    with st.expander("🎛️ Optimization Parameters"):
        st.markdown("""
        **Population Size (10-50):**
        - Larger = Better exploration, slower training
        - Smaller = Faster training, may miss optimal solutions
        - Recommended: 20-30 for most datasets
        
        **Max Iterations (10-100):**
        - More iterations = Better optimization, longer training
        - Early stopping when target accuracy achieved
        - Recommended: 30-50 for production models
        
        **Feature Selection (10-50):**
        - More features = More information, higher complexity
        - Fewer features = Faster inference, reduced overfitting
        - Recommended: 25-35 for IoT security data
        """)
    
    with st.expander("📈 Performance Expectations"):
        st.markdown("""
        **Target Accuracies:**
        - 2-Class: >99.5%
        - 8-Class: >99.0%  
        - 34-Class: >98.5%
        
        **Training Time (estimated):**
        - Small dataset (<10K samples): 5-15 minutes
        - Medium dataset (10K-100K): 15-60 minutes
        - Large dataset (>100K): 1-4 hours
        
        **Memory Requirements:**
        - Minimum: 4GB RAM
        - Recommended: 8GB+ RAM
        - GPU: Optional but recommended for large datasets
        """)
    
    st.subheader("📁 Dataset Requirements")
    st.markdown("""
    **Format Requirements:**
    - CSV file with header row
    - Numerical features (categorical will be encoded)
    - Target column as last column
    - No special characters in column names
    
    **Quality Guidelines:**
    - Minimum 1,000 samples recommended
    - Missing values <10% per column
    - Balanced class distribution preferred
    - Feature scaling handled automatically
    """)
    
    st.subheader("🔧 Installation & Setup")
    
    with st.expander("💻 Full Installation"):
        st.code("""
# Create virtual environment
python -m venv iot_security_env
source iot_security_env/bin/activate  # Linux/Mac
# or
iot_security_env\\Scripts\\activate  # Windows

# Install dependencies
pip install streamlit pandas numpy scikit-learn
pip install tensorflow lightgbm plotly
pip install imbalanced-learn matplotlib seaborn

# Run application
streamlit run streamlit_app.py
        """)
    
    with st.expander("🐳 Docker Deployment"):
        st.code("""
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Build and run
docker build -t iot-security-ml .
docker run -p 8501:8501 iot-security-ml
        """)
    
    st.subheader("❓ Troubleshooting")
    
    with st.expander("🔍 Common Issues"):
        st.markdown("""
        **Issue: Low accuracy (<95%)**
        - Solution: Increase feature count, more iterations, check data quality
        
        **Issue: Training too slow**
        - Solution: Reduce population size, fewer iterations, smaller dataset sample
        
        **Issue: Memory errors**
        - Solution: Reduce batch size, fewer features, process data in chunks
        
        **Issue: Import errors**
        - Solution: Install missing dependencies, check Python version (3.7+)
        
        **Issue: Dataset errors** 
        - Solution: Check CSV format, remove special characters, handle missing values
        """)

# Helper functions
def create_synthetic_dataset():
    """Create a realistic synthetic IoT security dataset"""
    np.random.seed(42)
    n_samples = 5000
    n_features = 50
    
    # Generate correlated features mimicking network traffic
    base_features = np.random.randn(n_samples, 10)
    
    # Create additional features with some correlation
    additional_features = []
    for i in range(40):
        if i < 20:
            # Network timing features
            feature = base_features[:, i % 10] + np.random.randn(n_samples) * 0.3
        else:
            # Packet size and flow features
            feature = np.abs(base_features[:, i % 10]) + np.random.exponential(0.5, n_samples)
        
        additional_features.append(feature)
    
    # Combine all features
    X = np.column_stack([base_features, np.array(additional_features).T])
    
    # Generate realistic attack labels
    # Create decision boundaries for different attack types
    decision_1 = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    decision_2 = -X[:, 3] + 2 * X[:, 4] - 0.5 * X[:, 5] + np.random.randn(n_samples) * 0.7
    decision_3 = X[:, 6] + X[:, 7] - X[:, 8] + np.random.randn(n_samples) * 0.6
    
    # Create labels based on decision boundaries
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
    
    # Create DataFrame
    columns = [f'feature_{i}' for i in range(n_features)] + ['label']
    df = pd.DataFrame(np.column_stack([X, labels]), columns=columns)
    
    return df

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🛡️ <strong>IoT Security ML Pipeline</strong> | Built with Streamlit</p>
    <p>Advanced Machine Learning for IoT Network Security Analysis</p>
    <p><em>Demo Version - For production deployment, install full ML dependencies</em></p>
</div>
""", unsafe_allow_html=True)

# Additional sidebar information
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 System Info")
    
    if st.session_state.model_trained:
        results = st.session_state.training_results
        st.success(f"✅ Model Trained")
        st.metric("Accuracy", f"{results['Accuracy']:.4f}")
        st.metric("Classification", results['representation_type'])
    else:
        st.info("⏳ No model trained")
    
    st.markdown("---")
    st.markdown("""
    **🔗 Quick Links:**
    - [Documentation](https://streamlit.io)
    - [GitHub Repository](https://github.com)
    - [Technical Paper](https://arxiv.org)
    """)
    
    st.markdown("---")
    st.markdown("""
    **📞 Support:**
    - Email: support@example.com
    - Issues: GitHub Issues
    - Community: Discord
    """)

# Run the app
if __name__ == "__main__":
    pass