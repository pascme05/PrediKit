# app.py - WITH PROPER FEATURE HANDLING
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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import traceback
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

data_store = {}

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
        
    def _handle_missing_values(self, df, name="data"):
        """Handle missing values in dataframe"""
        print(f"Checking for missing values in {name}...")
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"Found {nan_count} missing values in {name}")
            
            # Get columns with NaN
            nan_columns = df.columns[df.isna().any()].tolist()
            print(f"Columns with NaN: {nan_columns}")
            
            # Fill with mean for numerical columns
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    if df[col].isna().any():
                        mean_val = df[col].mean()
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
        
        # Check for infinite values
        if isinstance(X, pd.DataFrame):
            # Check for inf values
            inf_count = X.isin([np.inf, -np.inf]).sum().sum()
            if inf_count > 0:
                print(f"Found {inf_count} infinite values in {name}")
                X = X.replace([np.inf, -np.inf], np.nan)
                X = self._handle_missing_values(X, name)
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"Converting object column {col} to numeric...")
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if X[col].isna().any():
                        X[col].fillna(X[col].mean(), inplace=True)
                except:
                    print(f"Warning: Could not convert {col} to numeric, using label encoding")
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
        
        if y is not None:
            # Handle missing values in target
            if isinstance(y, pd.Series):
                if y.isna().any():
                    print(f"Found missing values in target, dropping rows with NaN target")
                    valid_idx = ~y.isna()
                    X = X[valid_idx]
                    y = y[valid_idx]
        
        return X, y
    
    def load_data(self, xlsx_file):
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
            
            # Load data and handle missing values
            train_df = pd.read_excel(xlsx_file, sheet_name='Train')
            train_df = self._handle_missing_values(train_df, "Train")
            
            val_df = pd.read_excel(xlsx_file, sheet_name='Val')
            val_df = self._handle_missing_values(val_df, "Val")
            
            test_df = pd.read_excel(xlsx_file, sheet_name='Test')
            test_df = self._handle_missing_values(test_df, "Test")
            
            print(f"Train shape: {train_df.shape}")
            print(f"Val shape: {val_df.shape}")
            print(f"Test shape: {test_df.shape}")
            
            for df, name in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
                if df.shape[1] < 3:
                    raise ValueError(f"{name} sheet must have at least 3 columns (Sample ID, features, target)")
            
            # Get column names
            self.sample_id_col = train_df.columns[0]
            print(f"Sample ID column: {self.sample_id_col}")
            
            # Extract feature columns (all except first and last)
            feature_cols = train_df.columns[1:-1].tolist()
            self.target_name = train_df.columns[-1]
            
            print(f"Feature columns: {feature_cols}")
            print(f"Target column: {self.target_name}")
            
            # Extract features and target using column names
            X_train = train_df[feature_cols].copy()
            y_train = train_df[self.target_name].copy()
            
            X_val = val_df[feature_cols].copy()
            y_val = val_df[self.target_name].copy()
            
            # Handle test data
            test_cols = test_df.columns.tolist()
            if len(test_cols) > len(feature_cols) + 1:
                # Test has target column
                self.has_test_labels = True
                X_test = test_df[feature_cols].copy()
                y_test = test_df[self.target_name].copy()
                test_sample_ids = test_df[self.sample_id_col].copy()
                
                # Check if target column is empty
                if y_test.isna().all() or (y_test.dtype == 'object' and y_test.str.strip().eq('').all()):
                    self.test_is_empty = True
                    self.has_test_labels = False
                    print(f"Test target column '{self.target_name}' is empty - treating as no labels")
                else:
                    self.test_is_empty = False
                    print(f"Test has labels in column '{self.target_name}'")
            else:
                # Test has no target column
                self.has_test_labels = False
                self.test_is_empty = True
                X_test = test_df[feature_cols].copy()
                y_test = None
                test_sample_ids = test_df[self.sample_id_col].copy()
                print("Test has no target column")
            
            # Store feature names
            self.feature_names = feature_cols
            
            # Validate and clean data
            X_train, y_train = self._validate_data(X_train, y_train, "Train")
            X_val, y_val = self._validate_data(X_val, y_val, "Val")
            X_test, _ = self._validate_data(X_test, None, "Test")
            
            print(f"X_train columns: {X_train.columns.tolist()}")
            print(f"X_val columns: {X_val.columns.tolist()}")
            print(f"X_test columns: {X_test.columns.tolist()}")
            
            # Determine task type
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
            
            # Final validation - ensure no NaN in training data
            print("\nFinal validation - checking for any remaining NaN values...")
            print(f"X_train NaN count: {X_train.isna().sum().sum()}")
            print(f"X_val NaN count: {X_val.isna().sum().sum()}")
            print(f"X_test NaN count: {X_test.isna().sum().sum()}")
            
            # If any NaN found, do aggressive cleaning
            if X_train.isna().sum().sum() > 0:
                print("WARNING: NaN values still present in X_train! Doing aggressive cleaning...")
                X_train = X_train.fillna(0)
                X_val = X_val.fillna(0)
                X_test = X_test.fillna(0)
            
            # Store data
            data_store['X_train'] = X_train
            data_store['X_val'] = X_val
            data_store['X_test'] = X_test
            data_store['y_train'] = y_train
            data_store['y_val'] = y_val
            data_store['y_test'] = y_test
            data_store['train_sample_ids'] = train_df[self.sample_id_col].copy()
            data_store['val_sample_ids'] = val_df[self.sample_id_col].copy()
            data_store['test_sample_ids'] = test_sample_ids
            data_store['feature_names'] = self.feature_names
            data_store['target_name'] = self.target_name
            data_store['task_type'] = self.task_type
            data_store['has_test_labels'] = self.has_test_labels
            
            print("=== DATA LOADED SUCCESSFULLY ===")
            
            return {
                'samples_train': len(X_train),
                'samples_val': len(X_val),
                'samples_test': len(X_test),
                'features': X_train.shape[1],
                'task_type': self.task_type,
                'has_test_labels': self.has_test_labels,
                'test_is_empty': self.test_is_empty,
                'feature_names': self.feature_names,
                'target_name': self.target_name
            }
        except Exception as e:
            print(f"ERROR in load_data: {str(e)}")
            traceback.print_exc()
            raise
    
    def _process_dnn_params(self, params):
        """Process DNN parameters to correct types"""
        processed_params = params.copy()
        
        if 'hidden_layer_sizes' in processed_params:
            hidden = processed_params['hidden_layer_sizes']
            if isinstance(hidden, str):
                try:
                    layers = [int(x.strip()) for x in hidden.split(',') if x.strip()]
                    if len(layers) == 1:
                        processed_params['hidden_layer_sizes'] = layers[0]
                    else:
                        processed_params['hidden_layer_sizes'] = tuple(layers)
                    print(f"Processed hidden_layer_sizes: {processed_params['hidden_layer_sizes']}")
                except ValueError as e:
                    print(f"Error parsing hidden_layer_sizes: {e}")
                    processed_params['hidden_layer_sizes'] = (100, 50)
            elif isinstance(hidden, (int, float)):
                processed_params['hidden_layer_sizes'] = int(hidden)
            elif isinstance(hidden, (list, tuple)):
                processed_params['hidden_layer_sizes'] = tuple(hidden)
        
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
            
            # Process DNN parameters if needed
            if model_name == 'DNN':
                params = self._process_dnn_params(params)
                print(f"Processed DNN Params: {params}")
            
            # Get the correct model class
            model_key = 'class' if self.task_type == 'classification' else 'reg'
            print(f"Looking for model with key: {model_key}")
            
            model_class = self.models[model_name][model_key]
            print(f"Model class: {model_class}")
            
            self.model = model_class(**params)
            print("Model instantiated")
            
            # Ensure no NaN in features before scaling
            print("Checking for NaN before scaling...")
            X_train = data_store['X_train']
            X_val = data_store['X_val']
            
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
            
            # Check for NaN after scaling
            if np.isnan(X_train_scaled).any():
                print("WARNING: NaN found after scaling! Replacing with 0")
                X_train_scaled = np.nan_to_num(X_train_scaled)
            
            if np.isnan(X_val_scaled).any():
                print("WARNING: NaN found after scaling! Replacing with 0")
                X_val_scaled = np.nan_to_num(X_val_scaled)
            
            print("Training model...")
            self.model.fit(X_train_scaled, data_store['y_train'])
            print("Model trained")
            
            print("Calculating scores...")
            train_score = self.model.score(X_train_scaled, data_store['y_train'])
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
            
            # Check for NaN in predictions
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
            
            # Determine which dataset to use for evaluation
            use_test = data_store['has_test_labels'] and data_store['y_test'] is not None
            
            if use_test:
                print("Using TEST set for evaluation")
                X_eval = data_store['X_test']
                y_true = data_store['y_test']
                eval_type = "test"
            else:
                print("Using VALIDATION set for evaluation (test has no labels)")
                X_eval = data_store['X_val']
                y_true = data_store['y_val']
                eval_type = "validation"
            
            # Ensure no NaN
            if X_eval.isna().sum().sum() > 0:
                print(f"WARNING: Found NaN in {eval_type} data, filling with 0")
                X_eval = X_eval.fillna(0)
            
            X_eval_scaled = self.scaler.transform(X_eval)
            
            # Check for NaN after scaling
            if np.isnan(X_eval_scaled).any():
                print("WARNING: NaN found in scaled data, replacing with 0")
                X_eval_scaled = np.nan_to_num(X_eval_scaled)
            
            y_pred = self.model.predict(X_eval_scaled)
            
            # Store predictions
            if use_test:
                data_store['test_predictions'] = y_pred
            else:
                data_store['test_predictions'] = y_pred
                data_store['test_sample_ids'] = data_store['val_sample_ids']
            
            if self.task_type == 'classification':
                accuracy = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix - {eval_type.capitalize()} Set')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                conf_matrix_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
                return {
                    'has_labels': True,
                    'eval_type': eval_type,
                    'accuracy': accuracy,
                    'classification_report': report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'confusion_matrix_img': conf_matrix_img,
                    'predictions': y_pred.tolist()
                }
            else:  # regression
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_true, y_pred, alpha=0.6)
                ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'Predictions vs Actual - {eval_type.capitalize()} Set')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                scatter_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
                residuals = y_true - y_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_pred, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title(f'Residual Plot - {eval_type.capitalize()} Set')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                residuals_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
                return {
                    'has_labels': True,
                    'eval_type': eval_type,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'scatter_img': scatter_img,
                    'residuals_img': residuals_img,
                    'predictions': y_pred.tolist()
                }
        except Exception as e:
            print(f"ERROR in evaluate_model: {str(e)}")
            traceback.print_exc()
            raise
    
    def predict_single(self, X_new):
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        # Ensure columns match what was trained on
        if hasattr(X_new, 'columns'):
            # Reorder columns to match training
            X_new = X_new[self.feature_names]
        elif isinstance(X_new, np.ndarray):
            # If numpy array, just use as is
            pass
        else:
            # Convert to DataFrame with correct columns
            X_new = pd.DataFrame(X_new, columns=self.feature_names)
        
        # Handle any NaN in input
        if hasattr(X_new, 'isna') and X_new.isna().sum().sum() > 0:
            print("WARNING: NaN in prediction input, filling with 0")
            X_new = X_new.fillna(0)
        
        X_scaled = self.scaler.transform(X_new)
        
        # Handle any NaN after scaling
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
        
        # Use test sample IDs if available and test has labels
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
            eval_type = "validation"
        
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
        
        data_info = trainer.load_data(filepath)
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'feature_names': trainer.feature_names,
            'target_name': trainer.target_name,
            'has_test_labels': trainer.has_test_labels,
            'test_is_empty': trainer.test_is_empty
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
        
        # Convert params to appropriate types
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
        return jsonify({
            'success': True,
            'results': results,
            'task_type': trainer.task_type,
            'has_test_labels': trainer.has_test_labels
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
        # Create DataFrame with correct feature names
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