import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GazelleOptimizationAlgorithm:
    """
    Enhanced GOA with feature selection capability for CNN+LightGBM
    """
    
    def __init__(self, population_size=25, max_iterations=40):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.best_position = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
    def initialize_population(self, bounds):
        """Initialize population within given bounds"""
        dim = len(bounds)
        population = np.zeros((self.population_size, dim))
        for i in range(dim):
            lower, upper = bounds[i]
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        return population
    
    def create_cnn_model(self, input_shape, num_classes, hyperparams):
        """
        Create CNN model with optimized architecture
        """
        try:
            # Extract CNN hyperparameters
            conv1_filters = int(hyperparams[0])
            conv2_filters = int(hyperparams[1])
            conv3_filters = int(hyperparams[2])
            dense1_units = int(hyperparams[3])
            dense2_units = int(hyperparams[4])
            dropout_rate = hyperparams[5]
            learning_rate = hyperparams[6]
            
            model = keras.Sequential([
                layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
                
                # First Conv Block
                layers.Conv1D(conv1_filters, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Dropout(dropout_rate * 0.5),
                
                # Second Conv Block
                layers.Conv1D(conv2_filters, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Dropout(dropout_rate * 0.7),
                
                # Third Conv Block
                layers.Conv1D(conv3_filters, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling1D(),
                
                # Dense layers
                layers.Dense(dense1_units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                
                layers.Dense(dense2_units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate * 0.8),
                
                # Output layer
                layers.Dense(num_classes, activation='softmax')
            ])
            
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            print(f"Error creating CNN model: {str(e)}")
            return None
    
    def fitness_function(self, position, X_train, y_train, X_val, y_val, n_features_to_select=30):
        """
        Enhanced fitness function for CNN+LightGBM with feature selection
        """
        try:
            # Extract hyperparameters (first 15 elements)
            cnn_hyperparams = position[:7]   # CNN parameters
            lgb_hyperparams = position[7:15] # LightGBM parameters
            
            # Extract feature weights
            feature_weights = position[15:]
            n_total_features = X_train.shape[1]
            
            # Validate feature weights dimension
            if len(feature_weights) != n_total_features:
                return float('inf')
                
            # Select top features
            if n_total_features > n_features_to_select:
                top_feature_indices = np.argsort(feature_weights)[-n_features_to_select:]
                X_train_reduced = X_train[:, top_feature_indices]
                X_val_reduced = X_val[:, top_feature_indices]
            else:
                X_train_reduced = X_train
                X_val_reduced = X_val
                
            num_classes = len(np.unique(y_train))
            input_shape = X_train_reduced.shape[1]
            
            # Create and train CNN model
            cnn_model = self.create_cnn_model(input_shape, num_classes, cnn_hyperparams)
            if cnn_model is None:
                return float('inf')
            
            # Train CNN with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=5, restore_best_weights=True
            )
            
            history = cnn_model.fit(
                X_train_reduced, y_train,
                validation_data=(X_val_reduced, y_val),
                epochs=20,
                batch_size=min(64, len(X_train_reduced) // 10),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Get CNN predictions (features)
            cnn_train_features = cnn_model.predict(X_train_reduced, verbose=0)
            cnn_val_features = cnn_model.predict(X_val_reduced, verbose=0)
            
            # Decode LightGBM hyperparameters
            lgb_params = {
                'objective': 'multiclass',
                'num_class': num_classes,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': int(lgb_hyperparams[0]),
                'learning_rate': lgb_hyperparams[1],
                'feature_fraction': lgb_hyperparams[2],
                'bagging_fraction': lgb_hyperparams[3],
                'bagging_freq': int(lgb_hyperparams[4]),
                'min_child_samples': int(lgb_hyperparams[5]),
                'lambda_l1': lgb_hyperparams[6],
                'lambda_l2': lgb_hyperparams[7],
                'verbose': -1
            }
            
            # Combine original features with CNN features
            X_train_combined = np.hstack([X_train_reduced, cnn_train_features])
            X_val_combined = np.hstack([X_val_reduced, cnn_val_features])
            
            # Train LightGBM
            train_data = lgb.Dataset(X_train_combined, label=y_train)
            val_data = lgb.Dataset(X_val_combined, label=y_val, reference=train_data)
            
            lgb_model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Make predictions
            lgb_pred_proba = lgb_model.predict(X_val_combined, num_iteration=lgb_model.best_iteration)
            lgb_pred = np.argmax(lgb_pred_proba, axis=1)
            
            # Calculate ensemble weight
            ensemble_weight = position[14]  # Weight for CNN vs LightGBM
            
            # Get CNN direct predictions
            cnn_pred_proba = cnn_model.predict(X_val_reduced, verbose=0)
            cnn_pred = np.argmax(cnn_pred_proba, axis=1)
            
            # Ensemble prediction
            final_pred_proba = ensemble_weight * cnn_pred_proba + (1 - ensemble_weight) * lgb_pred_proba
            final_pred = np.argmax(final_pred_proba, axis=1)
            
            accuracy = accuracy_score(y_val, final_pred)
            
            # Clear memory
            del cnn_model
            tf.keras.backend.clear_session()
            
            return 1 - accuracy  # Minimize error
            
        except Exception as e:
            print(f"Error in fitness function: {str(e)[:100]}")
            tf.keras.backend.clear_session()
            return float('inf')
    
    def optimize(self, X_train, y_train, X_val, y_val, bounds, n_features_to_select=30):
        """
        Main optimization loop with enhanced feature selection
        """
        print("Starting Gazelle Optimization for CNN+LightGBM...")
        population = self.initialize_population(bounds)
        fitness_values = np.zeros(self.population_size)
        dim = len(bounds)
        
        # Evaluate initial population
        print("Evaluating initial population...")
        for i in range(self.population_size):
            fitness_values[i] = self.fitness_function(
                population[i], X_train, y_train, X_val, y_val, n_features_to_select
            )
            if fitness_values[i] < self.best_fitness:
                self.best_fitness = fitness_values[i]
                self.best_position = population[i].copy()
        
        self.fitness_history.append(self.best_fitness)
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration+1}/{self.max_iterations} | Best Fitness: {self.best_fitness:.6f} | Accuracy: {1-self.best_fitness:.6f}")
            
            for i in range(self.population_size):
                # Enhanced exploration/exploitation with adaptive parameters
                exploration_prob = 0.7 * (1 - iteration / self.max_iterations)  # Decrease over time
                
                if np.random.rand() < exploration_prob:
                    # Exploration: Levy flight-inspired movement
                    r1, r2 = np.random.choice(self.population_size, 2, replace=False)
                    levy_step = np.random.normal(0, 1) * (iteration / self.max_iterations + 0.1)
                    population[i] += levy_step * (population[r1] - population[r2])
                else:
                    # Exploitation: Move towards best solution with perturbation
                    perturbation = 0.1 * (1 - iteration / self.max_iterations)
                    step = np.random.normal(0, perturbation)
                    population[i] += step * (self.best_position - population[i])
                
                # Apply bounds with reflection
                for d in range(dim):
                    low, high = bounds[d]
                    if population[i, d] < low:
                        population[i, d] = low + (low - population[i, d])
                    elif population[i, d] > high:
                        population[i, d] = high - (population[i, d] - high)
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
                        print(f"  → New best accuracy: {1-self.best_fitness:.6f}")
            
            self.fitness_history.append(self.best_fitness)
            
            # Early stopping if very high accuracy achieved
            if self.best_fitness < 0.001:  # 99.9% accuracy
                print(f"Early stopping: Achieved {1-self.best_fitness:.6f} accuracy")
                break
        
        print(f"Optimization completed. Final accuracy: {1-self.best_fitness:.6f}")
        return self.best_position

class CICIoT2023MLAlgorithm:
    """
    Enhanced CNN+LightGBM with feature selection using GOA
    """
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.output_scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.cnn_model = None
        self.lgb_model = None
        self.best_params = None
        self.feature_mask = None
        self.results = {}
        self.ensemble_weight = 0.5
        self.optimization_history = []
        
    def load_and_preprocess_data(self, file_path):
        """Enhanced data loading with better preprocessing"""
        print("Loading dataset...")
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Display basic info
            print(f"Columns: {list(df.columns)}")
            
            # Identify target column
            if 'label' in df.columns:
                target_col = 'label'
            else:
                target_col = df.columns[-1]
                print(f"Assuming '{target_col}' is the target column")
            
            # Move target to end
            if target_col != df.columns[-1]:
                cols = [col for col in df.columns if col != target_col] + [target_col]
                df = df[cols]
            
            print(f"Target column: {target_col}")
            print(f"Target distribution: {df[target_col].value_counts()}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e
        
        # Enhanced preprocessing
        df = self.handle_missing_values(df)
        df = self.handle_outliers_advanced(df)
        df = self.feature_engineering(df)
        
        return df
    
    def handle_missing_values(self, df):
        """Enhanced missing value handling"""
        print("Handling missing values and duplicates...")
        
        initial_shape = df.shape
        df = df.drop_duplicates()
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        target_col = df.columns[-1]
        
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if column == target_col:
                    df = df.dropna(subset=[column])
                elif df[column].dtype in ['float64', 'int64']:
                    # Use median for skewed data, mean for normal data
                    if abs(df[column].skew()) > 1:
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        df[column].fillna(df[column].mean(), inplace=True)
                else:
                    mode_val = df[column].mode()
                    if len(mode_val) > 0:
                        df[column].fillna(mode_val[0], inplace=True)
                    else:
                        df[column].fillna('Unknown', inplace=True)
        
        print(f"Final dataset shape after cleaning: {df.shape}")
        return df
    
    def handle_outliers_advanced(self, df):
        """Advanced outlier handling with multiple methods"""
        print("Handling outliers with advanced methods...")
        
        target_col = df.columns[-1]
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col != target_col]
        
        for column in numerical_columns:
            # Use different methods based on data distribution
            if df[column].nunique() < 10:  # Likely categorical
                continue
                
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                # More conservative outlier bounds for better model performance
                lower_bound = Q1 - 2.0 * IQR
                upper_bound = Q3 + 2.0 * IQR
                
                # Cap outliers
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        
        return df
    
    def feature_engineering(self, df):
        """Add engineered features for better performance"""
        print("Engineering additional features...")
        
        target_col = df.columns[-1]
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col != target_col]
        
        # Select a few key features for engineering to avoid explosion
        if len(numerical_columns) > 5:
            # Use correlation with target to select features
            correlations = []
            target_encoded = pd.get_dummies(df[target_col])
            target_numeric = target_encoded.iloc[:, 0]  # Use first class as proxy
            
            for col in numerical_columns[:10]:  # Limit to first 10 for speed
                corr = abs(df[col].corr(target_numeric))
                correlations.append((col, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [col for col, _ in correlations[:5]]
            
            # Create interaction features
            for i, feat1 in enumerate(top_features[:3]):
                for feat2 in top_features[i+1:4]:
                    # Ratio feature
                    col_name = f"{feat1}_ratio_{feat2}"
                    df[col_name] = df[feat1] / (df[feat2] + 1e-8)
                    
                    # Product feature
                    col_name = f"{feat1}_mult_{feat2}"
                    df[col_name] = df[feat1] * df[feat2]
        
        print(f"Dataset shape after feature engineering: {df.shape}")
        return df
    
    def create_class_representation(self, df, representation_type='2-class'):
        """Enhanced class representation with better balancing"""
        print(f"Creating {representation_type} representation...")
        
        target_col = df.columns[-1]
        unique_classes = df[target_col].unique()
        print(f"Unique classes: {unique_classes}")
        print(f"Class counts: {df[target_col].value_counts()}")
        
        if representation_type == '2-class':
            df_copy = df.copy()
            
            # Identify normal class
            normal_labels = ['Normal', 'BENIGN', 'Benign', 'normal', 'benign']
            normal_class = None
            
            for label in normal_labels:
                if label in unique_classes:
                    normal_class = label
                    break
            
            if normal_class is None:
                class_counts = df[target_col].value_counts()
                for class_name in class_counts.index:
                    if any(keyword in str(class_name).lower() for keyword in ['normal', 'benign', 'legitimate']):
                        normal_class = class_name
                        break
                if normal_class is None:
                    normal_class = class_counts.index[0]
            
            # Create binary classification
            df_copy[target_col] = df_copy[target_col].apply(
                lambda x: 'Normal' if x == normal_class else 'Attack'
            )
            
            # Enhanced balancing
            df_balanced = self.advanced_balancing(df_copy, target_samples=10000)
            
        elif representation_type == '8-class':
            df_copy = df.copy()
            
            # Enhanced attack categorization
            attack_mapping = self.create_attack_mapping(unique_classes)
            
            for new_class, old_classes in attack_mapping.items():
                if old_classes:
                    df_copy[target_col] = df_copy[target_col].replace(old_classes, new_class)
            
            df_balanced = self.advanced_balancing(df_copy, target_samples=5000)
            
        elif representation_type == '34-class':
            df_copy = df.copy()
            df_balanced = self.advanced_balancing(df_copy, target_samples=2000)
        
        print(f"Final {representation_type} dataset shape: {df_balanced.shape}")
        print(f"Class distribution:\n{df_balanced[target_col].value_counts()}")
        
        return df_balanced
    
    def create_attack_mapping(self, unique_classes):
        """Create intelligent attack mapping"""
        normal_labels = ['Normal', 'BENIGN', 'Benign', 'normal', 'benign']
        normal_class = None
        
        for label in normal_labels:
            if label in unique_classes:
                normal_class = label
                break
        
        attack_mapping = {}
        if normal_class:
            attack_mapping['Normal'] = [normal_class]
        
        remaining_classes = [c for c in unique_classes if c != normal_class]
        
        attack_categories = {
            'DDoS': [],
            'DoS': [],
            'Recon': [],
            'Web': [],
            'BruteForce': [],
            'Spoofing': [],
            'Mirai': []
        }
        
        # Intelligent categorization
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
            elif any(keyword in class_name_lower for keyword in ['brute', 'force', 'password']):
                attack_categories['BruteForce'].append(class_name)
            elif any(keyword in class_name_lower for keyword in ['spoof', 'mitm', 'arp']):
                attack_categories['Spoofing'].append(class_name)
            elif 'mirai' in class_name_lower:
                attack_categories['Mirai'].append(class_name)
            else:
                # Distribute remaining classes
                min_category = min(attack_categories.keys(), key=lambda x: len(attack_categories[x]))
                attack_categories[min_category].append(class_name)
        
        return {**attack_mapping, **attack_categories}
    
    def advanced_balancing(self, df, target_samples):
        """Advanced dataset balancing with multiple techniques"""
        target_col = df.columns[-1]
        class_counts = df[target_col].value_counts()
        
        # Adaptive target samples based on class distribution
        min_samples = min(class_counts)
        max_samples = max(class_counts)
        
        if target_samples > max_samples:
            target_samples = max_samples
        
        print(f"Target samples per class: {target_samples}")
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Apply stratified undersampling and SMOTE
        balanced_dfs = []
        
        for class_label in y.unique():
            class_data = df[df[target_col] == class_label]
            
            if len(class_data) > target_samples:
                # Stratified undersampling
                class_data = class_data.sample(n=target_samples, random_state=42)
            elif len(class_data) < target_samples:
                # Oversample with noise
                n_to_generate = target_samples - len(class_data)
                oversampled = resample(
                    class_data, 
                    n_samples=n_to_generate, 
                    random_state=42, 
                    replace=True
                )
                class_data = pd.concat([class_data, oversampled], ignore_index=True)
            
            balanced_dfs.append(class_data)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        # Apply SMOTE for final balancing
        if len(df_balanced) < target_samples * len(y.unique()):
            X_bal = df_balanced.drop(target_col, axis=1)
            y_bal = df_balanced[target_col]
            
            try:
                le_temp = LabelEncoder()
                y_encoded = le_temp.fit_transform(y_bal)
                
                # Conservative SMOTE application
                min_class_size = min(np.bincount(y_encoded))
                k_neighbors = min(3, max(1, min_class_size - 1))
                
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_smote, y_smote = smote.fit_resample(X_bal, y_encoded)
                
                y_smote_decoded = le_temp.inverse_transform(y_smote)
                df_balanced = pd.concat([
                    pd.DataFrame(X_smote, columns=X_bal.columns),
                    pd.Series(y_smote_decoded, name=target_col)
                ], axis=1)
                
            except Exception as e:
                print(f"SMOTE failed: {e}, using current balanced data")
        
        # Shuffle the final dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df_balanced
    
    def normalize_features(self, X_train, X_val, X_test):
        """Enhanced feature normalization"""
        print("Normalizing features...")
        
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_hybrid_model(self, X_train, y_train, X_val, y_val, n_features_to_select=30):
        """Train CNN+LightGBM hybrid model with enhanced optimization"""
        print(f"Training CNN+LightGBM hybrid model with top {n_features_to_select} features...")
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        n_total_features = X_train.shape[1]
        num_classes = len(np.unique(y_train_enc))
        
        print(f"Original features: {n_total_features}, Classes: {num_classes}")
        
        # Enhanced bounds for better optimization
        bounds = [
            # CNN hyperparameters
            (32, 128),     # conv1_filters
            (64, 256),     # conv2_filters
            (128, 512),    # conv3_filters
            (64, 256),     # dense1_units
            (32, 128),     # dense2_units
            (0.1, 0.5),    # dropout_rate
            (0.0001, 0.01), # learning_rate
            
            # LightGBM hyperparameters
            (20, 100),     # num_leaves
            (0.01, 0.3),   # learning_rate
            (0.6, 1.0),    # feature_fraction
            (0.6, 1.0),    # bagging_fraction
            (3, 10),       # bagging_freq
            (5, 50),       # min_child_samples
            (0.0, 1.0),    # lambda_l1
            (0.0, 1.0),    # lambda_l2
            
            (0.3, 0.7),    # ensemble_weight
        ] + [(0, 1)] * n_total_features  # Feature weights
        
        # Run optimization
        goa = GazelleOptimizationAlgorithm(population_size=20, max_iterations=35)
        best_position = goa.optimize(
            X_train, y_train_enc, X_val, y_val_enc, bounds, n_features_to_select
        )
        
        # Store results
        self.best_params = best_position
        self.optimization_history = goa.fitness_history
        
        # Extract optimized parameters
        cnn_params = best_position[:7]
        lgb_params = best_position[7:15]
        self.ensemble_weight = best_position[14]
        feature_weights = best_position[15:]
        
        # Create feature mask
        self.feature_mask = np.argsort(feature_weights)[-n_features_to_select:]
        print(f"Selected {len(self.feature_mask)} features from {n_total_features}")
        
        # Apply feature selection
        X_train_reduced = X_train[:, self.feature_mask]
        X_val_reduced = X_val[:, self.feature_mask]
        
        # Build final CNN model
        print("Building optimized CNN model...")
        self.cnn_model = self.create_final_cnn_model(
            X_train_reduced.shape[1], num_classes, cnn_params
        )
        
        # Train CNN with advanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_cnn_model.h5', save_best_only=True, monitor='val_accuracy'
            )
        ]
        
        print("Training CNN...")
        history = self.cnn