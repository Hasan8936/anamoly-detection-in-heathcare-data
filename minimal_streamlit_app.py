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
    page_title="DIRA - IoT Security Intelligence",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional security dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Montserrat', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        border: 1px solid #2C5364;
    }
    
    .main-header h1 {
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    .main-header h3 {
        font-weight: 600;
        color: #8FE3CF;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #2C5364;
        margin: 0.8rem 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        font-size: 1rem;
        color: #4a5568;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-card h2 {
        font-size: 1.8rem;
        color: #2C5364;
        margin-bottom: 0.2rem;
        font-weight: 700;
    }
    
    .status-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b8dfc1;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #90caf9;
        font-weight: 500;
    }
    
    .code-block {
        background: #2C5364;
        color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        font-family: monospace;
        border-left: 4px solid #8FE3CF;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding-top: 12px;
        padding-bottom: 12px;
        font-weight: 600;
        border: 1px solid #e9ecef;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2C5364;
        color: white;
        border-bottom: 3px solid #8FE3CF;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0F2027 0%, #203A43 100%);
        color: white;
    }
    
    .sidebar-header {
        color: #8FE3CF;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 1.5rem;
    }
    
    .param-slider {
        color: #2C5364;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #2C5364 0%, #203A43 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .download-btn:hover {
        background: linear-gradient(135deg, #203A43 0%, #0F2027 100%);
        color: white;
    }
    
    .training-btn {
        background: linear-gradient(135deg, #0F2027 0%, #2C5364 100%);
        font-weight: 700;
        font-size: 1.1rem;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .training-btn:hover {
        background: linear-gradient(135deg, #2C5364 0%, #0F2027 100%);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }
    
    .cyber-border {
        border: 1px solid #2C5364;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .result-badge {
        background: linear-gradient(135deg, #2C5364 0%, #203A43 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.3rem;
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
            status_container.markdown("<div class='cyber-border'>🔄 <b>Preprocessing data...</b></div>", unsafe_allow_html=True)
            progress_bar.progress(10)
            
            # Simulate preprocessing delay
            import time
            time.sleep(1)
            
            # Step 2: Feature selection setup
            status_container.markdown("<div class='cyber-border'>🎯 <b>Setting up feature selection...</b></div>", unsafe_allow_html=True)
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Step 3: Initialize Gazelle Optimization
            status_container.markdown("<div class='cyber-border'>🦌 <b>Initializing Gazelle Optimization Algorithm...</b></div>", unsafe_allow_html=True)
            progress_bar.progress(30)
            time.sleep(0.5)
            
            # Simulate optimization iterations
            best_accuracy = 0.7
            target_accuracy = self._get_target_accuracy(representation_type)
            
            for iteration in range(max_iterations):
                progress = 30 + (iteration / max_iterations) * 60
                status_container.markdown(f"<div class='cyber-border'>🔧 <b>Optimization iteration {iteration+1}/{max_iterations}</b></div>", unsafe_allow_html=True)
                progress_bar.progress(int(progress))
                
                # Simulate improvement
                improvement = np.random.exponential(0.01)
                best_accuracy = min(target_accuracy + 0.005, best_accuracy + improvement)
                self.training_history.append(best_accuracy)
                
                time.sleep(0.1)  # Small delay for visualization
            
            # Final model training
            status_container.markdown("<div class='cyber-border'>🧠 <b>Training final CNN+LightGBM ensemble...</b></div>", unsafe_allow_html=True)
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
            status_container.markdown("<div class='cyber-border'>✅ <b>Training completed successfully!</b></div>", unsafe_allow_html=True)
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
    <h1>🔒 DIRA - IoT Security Intelligence</h1>
    <h3>Advanced Threat Classification & Network Defense System</h3>
    <p>Real-time AI-powered IoT Security Analytics with Multi-Layer Protection</p>
</div>
""", unsafe_allow_html=True)

# System status
st.markdown("""
<div class="status-warning">
    <strong>🔧 DEMO MODE ACTIVE</strong> - Simulated results for demonstration purposes. 
    Production deployment requires full ML dependency stack including TensorFlow and LightGBM.
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("""
<div class="sidebar-header">
    ⚙️ SYSTEM CONFIGURATION
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class="sidebar-header" style="font-size: 1.2rem;">
    🎯 CLASSIFICATION SETTINGS
</div>
""", unsafe_allow_html=True)

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

st.sidebar.markdown("""
<div class="sidebar-header" style="font-size: 1.2rem;">
    🔧 OPTIMIZATION PARAMETERS
</div>
""", unsafe_allow_html=True)

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
st.sidebar.markdown(f"""
<div class="feature-box">
<strong>ACTIVE CONFIGURATION:</strong><br>
• Classification: <span class="result-badge">{representation_type}</span><br>
• Features: <span class="result-badge">{n_features}</span><br>
• Population: <span class="result-badge">{population_size}</span><br>
• Iterations: <span class="result-badge">{max_iterations}</span>
</div>
""", unsafe_allow_html=True)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 DATA & TRAINING", 
    "📈 RESULTS & ANALYTICS", 
    "🔍 MODEL INTELLIGENCE",
    "📚 DOCUMENTATION"
])

# Tab 1: Training & Data
with tab1:
    st.header("📊 DATASET INGESTION & MODEL TRAINING")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="cyber-border">
            <h3 style="color: #2C5364; margin-top: 0;">UPLOAD SECURITY DATASET</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload your IoT security dataset"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, nrows=1000)  # Preview first 1000 rows
                
                st.success("✅ DATASET LOADED SUCCESSFULLY!")
                
                # Dataset statistics
                col1_stat, col2_stat, col3_stat = st.columns(3)
                with col1_stat:
                    st.metric("Rows (preview)", len(df), help="Number of rows in the dataset preview")
                with col2_stat:
                    st.metric("Columns", len(df.columns), help="Number of features in the dataset")
                with col3_stat:
                    target_col = df.columns[-1]
                    st.metric("Classes", df[target_col].nunique(), help="Number of target classes")
                
                # Dataset preview
                with st.expander("📋 DATASET PREVIEW", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Class distribution
                with st.expander("📊 CLASS DISTRIBUTION ANALYSIS", expanded=True):
                    class_counts = df[target_col].value_counts()
                    st.bar_chart(class_counts)
                    
                    # Show statistics
                    st.write("**CLASS STATISTICS:**")
                    for class_name, count in class_counts.items():
                        st.write(f"- **{class_name}**: {count} samples")
                
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    
    with col2:
        st.markdown("""
        <div class="cyber-border">
            <h3 style="color: #2C5364; margin-top: 0;">TRAINING CONTROL CENTER</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
        <strong>ADVANCED TRAINING FEATURES:</strong><br>
        • Gazelle Optimization Algorithm<br>
        • Automated feature selection<br>
        • CNN + LightGBM ensemble<br>
        • Multi-layer security intelligence<br>
        • Target: >99% classification accuracy
        </div>
        """, unsafe_allow_html=True)
        
        # Training button
        if st.button("🚀 INITIATE MODEL TRAINING", type="primary", use_container_width=True, key="train_btn"):
            if uploaded_file is not None:
                df_full = pd.read_csv(uploaded_file)
                results = st.session_state.algorithm.simulate_training(
                    df_full, representation_type, n_features, population_size, max_iterations
                )
                if results:
                    st.session_state.training_results = results
                    st.session_state.model_trained = True
                    st.success(f"🎉 TRAINING COMPLETED! Accuracy: {results['Accuracy']:.4f}")
                    st.rerun()
            else:
                st.warning("Please upload a dataset first")
    
    # Sample dataset option
    st.markdown("---")
    st.markdown("""
    <div class="cyber-border">
        <h3 style="color: #2C5364; margin-top: 0;">DEMO WITH SYNTHETIC DATA</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1_demo, col2_demo = st.columns([3, 1])
    
    with col1_demo:
        st.info("""
        **GENERATE SYNTHETIC IOT SECURITY DATASET:**
        - 5,000 samples with realistic network traffic features
        - Multiple attack categories (DDoS, Malware, Intrusion, etc.)
        - Balanced class distribution for optimal training
        - Realistic feature correlations and patterns
        """)
    
    with col2_demo:
        if st.button("🎲 RUN DEMO TRAINING", use_container_width=True, key="demo_btn"):
            # Create sample dataset
            sample_df = create_synthetic_dataset()
            results = st.session_state.algorithm.simulate_training(
                sample_df, representation_type, n_features, population_size, max_iterations
            )
            if results:
                st.session_state.training_results = results
                st.session_state.model_trained = True
                st.success(f"🎉 DEMO TRAINING COMPLETED! Accuracy: {results['Accuracy']:.4f}")
                st.rerun()

# Tab 2: Results & Analysis
with tab2:
    st.header("📈 PERFORMANCE ANALYTICS & RESULTS")
    
    if st.session_state.model_trained and st.session_state.training_results:
        results = st.session_state.training_results
        
        # Performance metrics
        st.markdown("""
        <div class="cyber-border">
            <h3 style="color: #2C5364; margin-top: 0;">MODEL PERFORMANCE METRICS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = results['Accuracy']
            st.markdown(f"""
            <div class="metric-card">
                <h3>CLASSIFICATION ACCURACY</h3>
                <h2>{accuracy:.4f}</h2>
                <p>{accuracy*100:.2f}% PRECISION</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            precision = results['Precision']
            st.markdown(f"""
            <div class="metric-card">
                <h3>PRECISION RATE</h3>
                <h2>{precision:.4f}</h2>
                <p>{precision*100:.2f}% RELIABILITY</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            recall = results['Recall']
            st.markdown(f"""
            <div class="metric-card">
                <h3>RECALL RATE</h3>
                <h2>{recall:.4f}</h2>
                <p>{recall*100:.2f}% COVERAGE</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            f1_score = results['F1-Score']
            st.markdown(f"""
            <div class="metric-card">
                <h3>F1-SCORE</h3>
                <h2>{f1_score:.4f}</h2>
                <p>BALANCED METRIC</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance status
        target_accuracy = 0.99
        if accuracy >= target_accuracy:
            st.markdown(f"""
            <div class="status-success">
                🎉 <strong>SECURITY TARGET ACHIEVED!</strong><br>
                Classification accuracy ({accuracy:.4f}) exceeds the target threshold of {target_accuracy}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-warning">
                ⚡ <strong>GOOD PERFORMANCE ACHIEVED!</strong><br>
                Accuracy: {accuracy:.4f} (Target: {target_accuracy})<br>
                Consider increasing iterations or adjusting parameters for higher accuracy.
            </div>
            """, unsafe_allow_html=True)
        
        # Training convergence
        st.markdown("""
        <div class="cyber-border">
            <h3 style="color: #2C5364; margin-top: 0;">OPTIMIZATION CONVERGENCE</h3>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown("""
        <div class="cyber-border">
            <h3 style="color: #2C5364; margin-top: 0;">FEATURE SELECTION INTELLIGENCE</h3>
        </div>
        """, unsafe_allow_html=True)
        
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
                    'Feature': [f'Feature_{i+1}' for i in top_indices],
                    'Importance': importance[top_indices]
                })
                
                st.bar_chart(feature_data.set_index('Feature'))
    
    else:
        st.info("🔄 No training results available. Please train a model first in the Training tab.")

# Tab 3: Model Details
with tab3:
    st.header("🔍 MODEL INTELLIGENCE & CONFIGURATION")
    
    if st.session_state.model_trained and hasattr(st.session_state.algorithm, 'model_params'):
        params = st.session_state.algorithm.model_params
        
        # Model architecture
        st.markdown("""
        <div class="cyber-border">
            <h3 style="color: #2C5364; margin-top: 0;">OPTIMIZED ARCHITECTURE</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**CNN PARAMETERS:**")
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
            st.markdown("**LIGHTGBM PARAMETERS:**")
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
        st.markdown("""
        <div class="cyber-border">
            <h3 style="color: #2C5364; margin-top: 0;">ENSEMBLE CONFIGURATION</h3>
        </div>
        """, unsafe_allow_html=True)
        
        ensemble_weight = params['ensemble_weight']
        lgb_weight = 1 - ensemble_weight
        
        st.markdown(f"""
        <div class="feature-box">
        <strong>ENSEMBLE WEIGHTS:</strong><br>
        • CNN Contribution: {ensemble_weight:.1%}<br>
        • LightGBM Contribution: {lgb_weight:.1%}<br><br>
        <strong>OPTIMIZATION METHOD:</strong> Gazelle Optimization Algorithm (GOA)<br>
        <strong>FEATURE SELECTION:</strong> Top-{n_features} most important features
        </div>
        """, unsafe_allow_html=True)
        
        # Export model configuration
        st.markdown("""
        <div class="cyber-border">
            <h3 style="color: #2C5364; margin-top: 0;">EXPORT CONFIGURATION</h3>
        </div>
        """, unsafe_allow_html=True)
        
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
                label="📄 DOWNLOAD MODEL CONFIGURATION",
                data=json_str,
                file_name=f"dira_model_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2_export:
            st.info("""
            **EXPORT INCLUDES:**
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
    st.header("📚 SYSTEM DOCUMENTATION")
    
    st.markdown("""
    <div class="cyber-border">
        <h3 style="color: #2C5364; margin-top: 0;">SYSTEM OVERVIEW</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **DIRA (Digital IoT Resilience Analytics)** is an advanced AI-powered security system designed for IoT network protection:
    
    **CORE TECHNOLOGIES:**
    - **Convolutional Neural Network (CNN)**: Captures complex patterns in network traffic data
    - **LightGBM**: Provides fast, accurate gradient boosting for classification
    - **Gazelle Optimization Algorithm (GOA)**: Optimizes both model hyperparameters and feature selection
    - **Ensemble Learning**: Combines strengths of both models for superior performance
    """)
    
    st.markdown("""
    <div class="cyber-border">
        <h3 style="color: #2C5364; margin-top: 0;">CLASSIFICATION TYPES</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>2-CLASS DETECTION</h3>
            <p>Normal vs Attack</p>
            <p>Binary classification</p>
            <p>Highest accuracy target</p>
            <p>Best for general detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>8-CLASS ANALYSIS</h3>
            <p>7 attack categories + normal</p>
            <p>DDoS, DoS, Recon, etc.</p>
            <p>Balanced complexity/accuracy</p>
            <p>Good for attack identification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>34-CLASS FORENSICS</h3>
            <p>Full attack taxonomy</p>
            <p>Detailed attack classification</p>
            <p>Most challenging</p>
            <p>Best for forensic analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional documentation content would go here...

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
    columns = [f'feature_{i+1}' for i in range(n_features)] + ['label']
    df = pd.DataFrame(np.column_stack([X, labels]), columns=columns)
    
    return df

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #2C5364; margin-top: 2rem;'>
    <p>🔒 <strong>DIRA - IoT Security Intelligence System</strong> | Advanced Threat Detection</p>
    <p>Multi-Layer AI Protection for IoT Network Infrastructure</p>
    <p><em>Demo Version - For production deployment, install full ML dependencies</em></p>
</div>
""", unsafe_allow_html=True)

# Additional sidebar information
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-header" style="font-size: 1.2rem;">
        📊 SYSTEM STATUS
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_trained:
        results = st.session_state.training_results
        st.success(f"✅ MODEL TRAINED")
        st.metric("Accuracy", f"{results['Accuracy']:.4f}")
        st.metric("Classification", results['representation_type'])
    else:
        st.info("⏳ AWAITING TRAINING")
    
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-header" style="font-size: 1.2rem;">
        🔗 QUICK LINKS
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - [Documentation](https://streamlit.io)
    - [GitHub Repository](https://github.com)
    - [Technical Paper](https://arxiv.org)
    """)
    
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-header" style="font-size: 1.2rem;">
        📞 SUPPORT
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - Email: support@dira-security.com
    - Issues: GitHub Issues
    - Community: Discord
    """)

# Run the app
if __name__ == "__main__":
    pass
