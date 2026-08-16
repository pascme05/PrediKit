# app.py - WITH CLEAR VALIDATION INDICATION FOR NO GROUND TRUTH
from flask import Flask, render_template, request, jsonify, send_file, make_response
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import traceback
import zipfile
import tempfile
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB for images

os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

data_store = {}

class ImageFeatureExtractor:
    """Extract features from images using pre-trained CNN models"""
    
    def __init__(self, model_name='resnet50'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.feature_dim = None
        self._load_model()
        
    def _load_model(self):
        """Load pre-trained model and transform"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        elif self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 512
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        elif self.model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 4096
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        self.model.eval()
        self.model.to(self.device)
        print(f"Loaded {self.model_name}. Feature dimension: {self.feature_dim}")
        
    def extract_features(self, image_path):
        """Extract features from a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.flatten().cpu().numpy()
            
            return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_features_batch(self, image_paths, progress_callback=None):
        """Extract features from multiple images"""
        features = []
        valid_paths = []
        
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i, len(image_paths))
            
            feat = self.extract_features(path)
            if feat is not None:
                features.append(feat)
                valid_paths.append(path)
        
        if not features:
            return np.array([]), []
        
        return np.array(features), valid_paths

class ModelTrainer:
    def __init__(self):
        self.models = {
            'DT': {
                'class': DecisionTreeClassifier,
                'reg': DecisionTreeRegressor
            },
            'RF': {
                'class': RandomForestClassifier,
                'reg': RandomForestRegressor
            },
            'SVM': {
                'class': SVC,
                'reg': SVR
            },
            'DNN': {
                'class': MLPClassifier,
                'reg': MLPRegressor
            }
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.task_type = None
        self.feature_names = None
        self.target_name = None
        self.sample_id_col = 'Sample ID'
        self.all_predictions = None
        self.training_history = []
        self.has_test_labels = False
        self.test_is_empty = False
        self.eval_type = 'test'  # 'test' or 'validation'
        self.eval_message = ''   # Message explaining why validation is used
        self.last_evaluation_results = None
        self.image_extractor = None
        self.is_image_task = False
        self.image_folder = None
        
    def _handle_missing_values(self, df, name="data"):
        """Handle missing values in dataframe"""
        print(f"Checking for missing values in {name}...")
        
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"Found {nan_count} missing values in {name}")
            nan_columns = df.columns[df.isna().any()].tolist()
            print(f"Columns with NaN: {nan_columns}")
            
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    if df[col].isna().any():
                        mean_val = df[col].mean()
                        if pd.isna(mean_val):
                            mean_val = 0
                        df[col].fillna(mean_val, inplace=True)
                        print(f"Filled NaN in {col} with mean: {mean_val:.4f}")
                elif df[col].dtype == 'object':
                    if df[col].isna().any():
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                        df[col].fillna(mode_val, inplace=True)
                        print(f"Filled NaN in {col} with mode: {mode_val}")
            
            print(f"Missing values handled in {name}")
        else:
            print(f"No missing values found in {name}")
        
        return df
    
    def _validate_data(self, X, y=None, name="data"):
        """Validate data and handle any issues"""
        print(f"Validating {name}...")
        
        if isinstance(X, pd.DataFrame):
            inf_count = X.isin([np.inf, -np.inf]).sum().sum()
            if inf_count > 0:
                print(f"Found {inf_count} infinite values in {name}")
                X = X.replace([np.inf, -np.inf], np.nan)
                X = self._handle_missing_values(X, name)
        
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"Converting object column {col} to numeric...")
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if X[col].isna().any():
                        X[col].fillna(X[col].mean() if not pd.isna(X[col].mean()) else 0, inplace=True)
                except:
                    print(f"Warning: Could not convert {col} to numeric, using label encoding")
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
        
        if isinstance(X, pd.DataFrame):
            if X.isna().sum().sum() > 0:
                print(f"WARNING: Found remaining NaN in {name}, filling with 0")
                X = X.fillna(0)
        
        if y is not None:
            if isinstance(y, pd.Series):
                if y.isna().any():
                    print(f"Found missing values in target, dropping rows with NaN target")
                    valid_idx = ~y.isna()
                    X = X[valid_idx]
                    y = y[valid_idx]
                if y.dtype == 'object':
                    try:
                        y = pd.to_numeric(y, errors='coerce')
                        if y.isna().any():
                            print(f"Warning: Could not convert target to numeric, using label encoding")
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y = le.fit_transform(y.astype(str))
                    except:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
        
        return X, y
    
    def _extract_image_features(self, df, image_folder):
        """Extract image features from dataframe with image names"""
        print("=== EXTRACTING IMAGE FEATURES ===")
        
        if self.image_extractor is None:
            self.image_extractor = ImageFeatureExtractor('resnet50')
        
        image_col = df.columns[1]
        image_names = df[image_col].tolist()
        
        image_paths = []
        valid_indices = []
        
        for i, name in enumerate(image_names):
            name_str = str(name).strip()
            found = False
            for ext in ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                path = os.path.join(image_folder, name_str + ext)
                if os.path.exists(path):
                    image_paths.append(path)
                    valid_indices.append(i)
                    found = True
                    break
            if not found:
                print(f"Warning: Image not found: {name_str}")
        
        if not image_paths:
            raise ValueError(f"No images found in {image_folder}. Check image names and folder.")
        
        print(f"Found {len(image_paths)} images out of {len(image_names)} samples")
        
        features, valid_paths = self.image_extractor.extract_features_batch(image_paths)
        
        if len(features) == 0:
            raise ValueError("Failed to extract features from any images")
        
        valid_df = df.iloc[valid_indices].copy()
        valid_df['image_path'] = valid_paths
        
        feature_names = [f'feat_{i}' for i in range(features.shape[1])]
        feature_df = pd.DataFrame(features, columns=feature_names, index=valid_df.index)
        
        result_df = pd.concat([valid_df[[df.columns[0], df.columns[-1]]], feature_df], axis=1)
        
        if result_df.isna().sum().sum() > 0:
            print("WARNING: NaN in extracted features, filling with 0")
            result_df = result_df.fillna(0)
        
        print(f"Extracted {features.shape[1]} features from {len(valid_paths)} images")
        
        return result_df, feature_names
    
    def load_data(self, xlsx_file, image_folder=None, model_name='resnet50'):
        """Load data from Excel file with three worksheets: Train, Val, Test"""
        try:
            print("=== LOADING DATA ===")
            xls = pd.ExcelFile(xlsx_file)
            sheet_names = xls.sheet_names
            print(f"Worksheets found: {sheet_names}")
            
            required_sheets = ['Train', 'Val', 'Test']
            for sheet in required_sheets:
                if sheet not in sheet_names:
                    raise ValueError(f"Required worksheet '{sheet}' not found. Found: {sheet_names}")
            
            train_df = pd.read_excel(xlsx_file, sheet_name='Train')
            train_df = self._handle_missing_values(train_df, "Train")
            
            val_df = pd.read_excel(xlsx_file, sheet_name='Val')
            val_df = self._handle_missing_values(val_df, "Val")
            
            test_df = pd.read_excel(xlsx_file, sheet_name='Test')
            test_df = self._handle_missing_values(test_df, "Test")
            
            print(f"Train shape: {train_df.shape}")
            print(f"Val shape: {val_df.shape}")
            print(f"Test shape: {test_df.shape}")
            
            self.is_image_task = image_folder is not None and os.path.exists(image_folder)
            
            if self.is_image_task:
                print(f"Processing as IMAGE CLASSIFICATION task. Image folder: {image_folder}")
                self.image_folder = image_folder
                
                train_df, feature_names = self._extract_image_features(train_df, image_folder)
                val_df, _ = self._extract_image_features(val_df, image_folder)
                test_df, _ = self._extract_image_features(test_df, image_folder)
                
                self.feature_names = feature_names
                self.sample_id_col = train_df.columns[0]
                self.target_name = train_df.columns[-1]
            else:
                self.sample_id_col = train_df.columns[0]
                feature_cols = train_df.columns[1:-1].tolist()
                self.feature_names = feature_cols
                self.target_name = train_df.columns[-1]
                
                print(f"Processing as TABULAR data. Features: {self.feature_names}")
            
            if self.is_image_task:
                X_train = train_df.iloc[:, 2:-1]
                y_train = train_df.iloc[:, -1]
                
                X_val = val_df.iloc[:, 2:-1]
                y_val = val_df.iloc[:, -1]
                
                test_cols = test_df.shape[1]
                if test_cols > 3:
                    self.has_test_labels = True
                    X_test = test_df.iloc[:, 2:-1]
                    y_test = test_df.iloc[:, -1]
                    test_sample_ids = test_df.iloc[:, 0]
                else:
                    self.has_test_labels = False
                    X_test = test_df.iloc[:, 2:]
                    y_test = None
                    test_sample_ids = test_df.iloc[:, 0]
            else:
                X_train = train_df.iloc[:, 1:-1]
                y_train = train_df.iloc[:, -1]
                
                X_val = val_df.iloc[:, 1:-1]
                y_val = val_df.iloc[:, -1]
                
                test_cols = test_df.shape[1]
                if test_cols > len(self.feature_names) + 1:
                    self.has_test_labels = True
                    X_test = test_df.iloc[:, 1:-1]
                    y_test = test_df.iloc[:, -1]
                    test_sample_ids = test_df.iloc[:, 0]
                else:
                    self.has_test_labels = False
                    X_test = test_df.iloc[:, 1:]
                    y_test = None
                    test_sample_ids = test_df.iloc[:, 0]
            
            X_train, y_train = self._validate_data(X_train, y_train, "Train")
            X_val, y_val = self._validate_data(X_val, y_val, "Val")
            X_test, _ = self._validate_data(X_test, None, "Test")
            
            print(f"X_train shape: {X_train.shape}")
            print(f"X_val shape: {X_val.shape}")
            print(f"X_test shape: {X_test.shape}")
            
            y_train_series = y_train
            if y_train_series.dtype == 'object' or len(y_train_series.unique()) < 10:
                self.task_type = 'classification'
                print("Task type: CLASSIFICATION")
                
                y_train_encoded = self.label_encoder.fit_transform(y_train_series)
                y_train = y_train_encoded
                print(f"Classes: {self.label_encoder.classes_}")
                
                if y_val is not None:
                    y_val_encoded = self.label_encoder.transform(y_val)
                    y_val = y_val_encoded
                
                if y_test is not None and self.has_test_labels:
                    try:
                        y_test_encoded = self.label_encoder.transform(y_test)
                        y_test = y_test_encoded
                    except Exception as e:
                        print(f"Error encoding test labels: {e}")
                        y_test_encoded = np.array([-1 if label not in self.label_encoder.classes_ else 
                                                   self.label_encoder.transform([label])[0] 
                                                   for label in y_test])
                        y_test = y_test_encoded
            else:
                self.task_type = 'regression'
                print("Task type: REGRESSION")
            
            # Aggressive cleaning for all data
            for name, data in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test)]:
                if data.isna().sum().sum() > 0:
                    print(f"WARNING: NaN in {name}, filling with 0")
                    data.fillna(0, inplace=True)
                    data_store[name] = data
            
            data_store['X_train'] = X_train
            data_store['X_val'] = X_val
            data_store['X_test'] = X_test
            data_store['y_train'] = y_train
            data_store['y_val'] = y_val
            data_store['y_test'] = y_test
            data_store['train_sample_ids'] = train_df.iloc[:, 0]
            data_store['val_sample_ids'] = val_df.iloc[:, 0]
            data_store['test_sample_ids'] = test_sample_ids
            data_store['feature_names'] = self.feature_names
            data_store['target_name'] = self.target_name
            data_store['task_type'] = self.task_type
            data_store['has_test_labels'] = self.has_test_labels
            data_store['is_image_task'] = self.is_image_task
            data_store['image_folder'] = self.image_folder
            
            # Set evaluation type message
            if not self.has_test_labels:
                self.eval_message = "Test set has no ground truth labels. Evaluation shown using Validation set."
            else:
                self.eval_message = "Evaluation using Test set."
            
            print("=== DATA LOADED SUCCESSFULLY ===")
            
            return {
                'samples_train': len(X_train),
                'samples_val': len(X_val),
                'samples_test': len(X_test),
                'features': X_train.shape[1],
                'task_type': self.task_type,
                'has_test_labels': self.has_test_labels,
                'is_image_task': self.is_image_task,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'eval_message': self.eval_message
            }
        except Exception as e:
            print(f"ERROR in load_data: {str(e)}")
            traceback.print_exc()
            raise
    
    def _process_dnn_params(self, params):
        """Process DNN parameters to correct types"""
        processed_params = {}
        
        defaults = {
            'hidden_layer_sizes': (100, 50),
            'max_iter': 200,
            'alpha': 0.0001,
            'learning_rate_init': 0.001,
            'random_state': 42
        }
        
        for key, default_value in defaults.items():
            processed_params[key] = default_value
        
        for key, value in params.items():
            if key == 'hidden_layer_sizes':
                if isinstance(value, str):
                    try:
                        layers = [int(x.strip()) for x in value.split(',') if x.strip()]
                        if len(layers) == 1:
                            processed_params['hidden_layer_sizes'] = layers[0]
                        else:
                            processed_params['hidden_layer_sizes'] = tuple(layers)
                        print(f"Parsed hidden_layer_sizes: {processed_params['hidden_layer_sizes']}")
                    except ValueError as e:
                        print(f"Error parsing hidden_layer_sizes: {e}, using default")
                        processed_params['hidden_layer_sizes'] = defaults['hidden_layer_sizes']
                elif isinstance(value, (int, float)):
                    processed_params['hidden_layer_sizes'] = int(value)
                elif isinstance(value, (list, tuple)):
                    processed_params['hidden_layer_sizes'] = tuple(value)
            elif key == 'alpha':
                try:
                    processed_params['alpha'] = float(value)
                except:
                    processed_params['alpha'] = defaults['alpha']
            elif key == 'max_iter':
                try:
                    processed_params['max_iter'] = int(value)
                except:
                    processed_params['max_iter'] = defaults['max_iter']
            else:
                processed_params[key] = value
        
        return processed_params
    
    def train_model(self, model_name, **params):
        """Train the selected model"""
        print("=== TRAINING MODEL ===")
        print(f"Model: {model_name}")
        print(f"Original Params: {params}")
        print(f"Task type: {self.task_type}")
        
        try:
            if self.task_type not in ['classification', 'regression']:
                raise ValueError(f"Task type not set. Current: {self.task_type}")
            
            if model_name == 'DNN':
                params = self._process_dnn_params(params)
                print(f"Processed DNN Params: {params}")
            
            model_key = 'class' if self.task_type == 'classification' else 'reg'
            print(f"Looking for model with key: {model_key}")
            
            model_class = self.models[model_name][model_key]
            print(f"Model class: {model_class}")
            
            if model_name == 'SVM' and self.task_type == 'classification':
                if 'probability' not in params:
                    params['probability'] = True
                if 'random_state' not in params:
                    params['random_state'] = 42
            
            self.model = model_class(**params)
            print("Model instantiated")
            
            X_train = data_store['X_train'].copy()
            X_val = data_store['X_val'].copy()
            
            if X_train.isna().sum().sum() > 0:
                print(f"WARNING: Found {X_train.isna().sum().sum()} NaN in X_train, filling with 0")
                X_train = X_train.fillna(0)
                data_store['X_train'] = X_train
            
            if X_val.isna().sum().sum() > 0:
                print(f"WARNING: Found {X_val.isna().sum().sum()} NaN in X_val, filling with 0")
                X_val = X_val.fillna(0)
                data_store['X_val'] = X_val
            
            print("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            print(f"X_train shape: {X_train_scaled.shape}")
            print(f"X_val shape: {X_val_scaled.shape}")
            
            if np.isnan(X_train_scaled).any():
                print("WARNING: NaN found after scaling! Replacing with 0")
                X_train_scaled = np.nan_to_num(X_train_scaled)
            
            if np.isnan(X_val_scaled).any():
                print("WARNING: NaN found after scaling! Replacing with 0")
                X_val_scaled = np.nan_to_num(X_val_scaled)
            
            y_train = data_store['y_train']
            if isinstance(y_train, pd.Series):
                if y_train.isna().any():
                    print("WARNING: NaN in y_train, filling with 0")
                    y_train = y_train.fillna(0)
                    data_store['y_train'] = y_train
            
            print("Training model...")
            self.model.fit(X_train_scaled, y_train)
            print("Model trained")
            
            print("Calculating scores...")
            train_score = self.model.score(X_train_scaled, y_train)
            val_score = self.model.score(X_val_scaled, data_store['y_val'])
            print(f"Train score: {train_score}")
            print(f"Val score: {val_score}")
            
            self.training_history.append({
                'model_name': model_name,
                'params': params,
                'train_score': train_score,
                'val_score': val_score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print("Making predictions on all data...")
            X_all = pd.concat([data_store['X_train'], data_store['X_val'], data_store['X_test']])
            X_all_scaled = self.scaler.transform(X_all)
            
            if np.isnan(X_all_scaled).any():
                print("WARNING: NaN found in X_all_scaled, replacing with 0")
                X_all_scaled = np.nan_to_num(X_all_scaled)
            
            self.all_predictions = self.model.predict(X_all_scaled)
            
            print("=== TRAINING COMPLETE ===")
            
            return {
                'train_score': train_score,
                'val_score': val_score,
                'model_name': model_name,
                'params': params
            }
        except Exception as e:
            print(f"ERROR in train_model: {str(e)}")
            traceback.print_exc()
            raise
    
    def evaluate_model(self):
        """Evaluate the trained model - uses test if available, otherwise validation"""
        print("=== EVALUATING MODEL ===")
        try:
            if self.model is None:
                raise ValueError("No model trained yet.")
            
            # Determine which dataset to use
            use_test = data_store['has_test_labels'] and data_store['y_test'] is not None
            
            if use_test:
                print("Using TEST set for evaluation")
                X_eval = data_store['X_test'].copy()
                y_true = data_store['y_test']
                self.eval_type = 'test'
                eval_message = "Evaluation using Test Set"
            else:
                print("Using VALIDATION set for evaluation (test has no labels)")
                X_eval = data_store['X_val'].copy()
                y_true = data_store['y_val']
                self.eval_type = 'validation'
                eval_message = "⚠️ Test set has no ground truth. Evaluation using Validation Set"
            
            # Clean evaluation data
            if X_eval.isna().sum().sum() > 0:
                print(f"WARNING: Found NaN in {self.eval_type} data, filling with 0")
                X_eval = X_eval.fillna(0)
            
            X_eval_scaled = self.scaler.transform(X_eval)
            
            if np.isnan(X_eval_scaled).any():
                print("WARNING: NaN found in scaled data, replacing with 0")
                X_eval_scaled = np.nan_to_num(X_eval_scaled)
            
            y_pred = self.model.predict(X_eval_scaled)
            
            if use_test:
                data_store['test_predictions'] = y_pred
                data_store['test_sample_ids'] = data_store['test_sample_ids']
            else:
                data_store['test_predictions'] = y_pred
                data_store['test_sample_ids'] = data_store['val_sample_ids']
            
            self.last_evaluation_results = {
                'eval_type': self.eval_type,
                'eval_message': eval_message,
                'predictions': y_pred.tolist(),
                'y_true': y_true.tolist()
            }
            
            if self.task_type == 'classification':
                accuracy = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix - {self.eval_type.capitalize()} Set')
                
                # Add subtitle if using validation
                if self.eval_type == 'validation':
                    plt.figtext(0.5, 0.01, '⚠️ Test set has no ground truth - using Validation set', 
                               ha='center', fontsize=10, color='orange')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                conf_matrix_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
                result = {
                    'has_labels': True,
                    'eval_type': self.eval_type,
                    'eval_message': eval_message,
                    'accuracy': accuracy,
                    'classification_report': report,
                    'confusion_matrix_img': conf_matrix_img,
                    'predictions': y_pred.tolist(),
                    'y_true': y_true.tolist()
                }
                self.last_evaluation_results.update(result)
                return result
                
            else:  # regression
                y_true_clean = np.nan_to_num(y_true)
                y_pred_clean = np.nan_to_num(y_pred)
                
                mse = mean_squared_error(y_true_clean, y_pred_clean)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                r2 = r2_score(y_true_clean, y_pred_clean)
                
                # Line plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sorted_idx = np.argsort(y_true_clean)
                y_true_sorted = y_true_clean[sorted_idx]
                y_pred_sorted = y_pred_clean[sorted_idx]
                
                ax.plot(range(len(y_true_sorted)), y_true_sorted, 'b-', label='Actual', linewidth=2)
                ax.plot(range(len(y_pred_sorted)), y_pred_sorted, 'r--', label='Predicted', linewidth=2)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Value')
                
                # Title with clear indication
                if self.eval_type == 'validation':
                    ax.set_title(f'Predictions vs Actual - Validation Set ⚠️ (Test has no ground truth)')
                else:
                    ax.set_title(f'Predictions vs Actual - {self.eval_type.capitalize()} Set')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add note if using validation
                if self.eval_type == 'validation':
                    plt.figtext(0.5, 0.01, '⚠️ Test set has no ground truth - using Validation set for evaluation', 
                               ha='center', fontsize=10, color='orange')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                line_plot_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
                # Scatter plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_true_clean, y_pred_clean, alpha=0.6)
                ax.plot([y_true_clean.min(), y_true_clean.max()], [y_true_clean.min(), y_true_clean.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                
                if self.eval_type == 'validation':
                    ax.set_title(f'Scatter Plot - Validation Set ⚠️ (Test has no ground truth)')
                else:
                    ax.set_title(f'Scatter Plot - {self.eval_type.capitalize()} Set')
                
                ax.grid(True, alpha=0.3)
                
                if self.eval_type == 'validation':
                    plt.figtext(0.5, 0.01, '⚠️ Test set has no ground truth - using Validation set', 
                               ha='center', fontsize=10, color='orange')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                scatter_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
                # Residual plot
                residuals = y_true_clean - y_pred_clean
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_pred_clean, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                
                if self.eval_type == 'validation':
                    ax.set_title(f'Residual Plot - Validation Set ⚠️ (Test has no ground truth)')
                else:
                    ax.set_title(f'Residual Plot - {self.eval_type.capitalize()} Set')
                
                ax.grid(True, alpha=0.3)
                
                if self.eval_type == 'validation':
                    plt.figtext(0.5, 0.01, '⚠️ Test set has no ground truth - using Validation set', 
                               ha='center', fontsize=10, color='orange')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                residuals_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
                result = {
                    'has_labels': True,
                    'eval_type': self.eval_type,
                    'eval_message': eval_message,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'line_plot_img': line_plot_img,
                    'scatter_img': scatter_img,
                    'residuals_img': residuals_img,
                    'predictions': y_pred_clean.tolist(),
                    'y_true': y_true_clean.tolist()
                }
                self.last_evaluation_results.update(result)
                return result
                
        except Exception as e:
            print(f"ERROR in evaluate_model: {str(e)}")
            traceback.print_exc()
            raise
    
    def predict_single(self, X_new):
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        if hasattr(X_new, 'columns'):
            X_new = X_new[self.feature_names]
        elif isinstance(X_new, np.ndarray):
            pass
        else:
            X_new = pd.DataFrame(X_new, columns=self.feature_names)
        
        if hasattr(X_new, 'isna') and X_new.isna().sum().sum() > 0:
            print("WARNING: NaN in prediction input, filling with 0")
            X_new = X_new.fillna(0)
        
        X_scaled = self.scaler.transform(X_new)
        
        if np.isnan(X_scaled).any():
            print("WARNING: NaN after scaling, replacing with 0")
            X_scaled = np.nan_to_num(X_scaled)
        
        predictions = self.model.predict(X_scaled)
        
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def download_predictions(self):
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        if data_store['has_test_labels'] and data_store['y_test'] is not None:
            sample_ids = data_store['test_sample_ids']
            predictions = data_store['test_predictions']
            has_actual = True
            actual = data_store['y_test']
            eval_type = "test"
        else:
            sample_ids = data_store['val_sample_ids']
            predictions = data_store['test_predictions']
            has_actual = True
            actual = data_store['y_val']
            eval_type = "validation ⚠️ (no test labels)"
        
        if self.task_type == 'classification':
            predictions_decoded = self.label_encoder.inverse_transform(predictions.astype(int))
            if has_actual and actual is not None:
                actual_decoded = self.label_encoder.inverse_transform(actual.astype(int))
        else:
            predictions_decoded = predictions
            if has_actual and actual is not None:
                actual_decoded = actual
        
        results_df = pd.DataFrame({
            'Sample ID': sample_ids,
            'Predictions': predictions_decoded
        })
        
        if has_actual and actual is not None:
            results_df['Actual'] = actual_decoded
            results_df['Evaluation_Set'] = eval_type
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            summary_data = {
                'Metric': ['Model', 'Task Type', 'Evaluation Set', 'Number of Predictions', 'Date Generated'],
                'Value': [
                    self.training_history[-1]['model_name'] if self.training_history else 'N/A',
                    self.task_type,
                    eval_type.capitalize(),
                    len(results_df),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output

trainer = ModelTrainer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'error': 'File must be Excel format (.xlsx or .xls)'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image_folder = None
        if 'images' in request.files:
            image_file = request.files['images']
            if image_file and image_file.filename != '':
                if image_file.filename.endswith('.zip'):
                    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
                    image_file.save(zip_path)
                    
                    image_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'images_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
                    os.makedirs(image_folder, exist_ok=True)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(image_folder)
                    
                    os.remove(zip_path)
                    print(f"Extracted images to {image_folder}")
                else:
                    return jsonify({'error': 'Images must be uploaded as a ZIP file'}), 400
        
        data_info = trainer.load_data(filepath, image_folder)
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'feature_names': trainer.feature_names,
            'target_name': trainer.target_name,
            'has_test_labels': trainer.has_test_labels,
            'test_is_empty': trainer.test_is_empty,
            'is_image_task': trainer.is_image_task,
            'eval_message': trainer.eval_message
        })
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    print("\n" + "="*50)
    print("TRAIN REQUEST RECEIVED")
    print("="*50)
    
    try:
        data = request.json
        print(f"Data: {data}")
        
        model_name = data.get('model_name')
        params = data.get('params', {})
        
        print(f"Model: {model_name}")
        print(f"Params: {params}")
        
        if not model_name:
            return jsonify({'error': 'Model name not specified'}), 400
        
        for key, value in params.items():
            if isinstance(value, str):
                if value.isdigit():
                    params[key] = int(value)
                elif value.replace('.', '').replace('-', '').isdigit():
                    try:
                        params[key] = float(value)
                    except:
                        pass
        
        print("Calling trainer.train_model...")
        results = trainer.train_model(model_name, **params)
        print(f"Results: {results}")
        
        response_data = {
            'success': True,
            'results': results,
            'task_type': trainer.task_type
        }
        print(f"Response: {response_data}")
        print("="*50 + "\n")
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"ERROR in train route: {str(e)}")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    try:
        results = trainer.evaluate_model()
        print(f"Evaluation results keys: {results.keys()}")
        return jsonify({
            'success': True,
            'results': results,
            'task_type': trainer.task_type,
            'has_test_labels': trainer.has_test_labels,
            'eval_message': trainer.eval_message
        })
    
    except Exception as e:
        print(f"Evaluate error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features')
    
    if not features:
        return jsonify({'error': 'No features provided'}), 400
    
    try:
        X_new = pd.DataFrame([features], columns=trainer.feature_names)
        predictions = trainer.predict_single(X_new)
        
        return jsonify({
            'success': True,
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
        })
    
    except Exception as e:
        print(f"Predict error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['GET'])
def download_predictions():
    try:
        excel_file = trainer.download_predictions()
        
        response = make_response(excel_file.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = f'attachment; filename=predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        return response
    
    except Exception as e:
        print(f"Download error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_data():
    global data_store
    data_store = {}
    trainer.model = None
    trainer.all_predictions = None
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)