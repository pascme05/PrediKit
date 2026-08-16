# app.py - CORRECTED VERSION
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
        # FIXED: Keys are 'class' for classification and 'reg' for regression
        self.models = {
            'DT': {
                'class': DecisionTreeClassifier,  # For classification
                'reg': DecisionTreeRegressor      # For regression
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
            
            train_df = pd.read_excel(xlsx_file, sheet_name='Train')
            val_df = pd.read_excel(xlsx_file, sheet_name='Val')
            test_df = pd.read_excel(xlsx_file, sheet_name='Test')
            
            print(f"Train shape: {train_df.shape}")
            print(f"Val shape: {val_df.shape}")
            print(f"Test shape: {test_df.shape}")
            
            for df, name in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
                if df.shape[1] < 3:
                    raise ValueError(f"{name} sheet must have at least 3 columns (Sample ID, features, target)")
            
            self.has_test_labels = test_df.shape[1] > 2
            print(f"Test has labels: {self.has_test_labels}")
            
            self.sample_id_col = train_df.columns[0]
            print(f"Sample ID column: {self.sample_id_col}")
            
            X_train = train_df.iloc[:, 1:-1]
            y_train = train_df.iloc[:, -1]
            
            X_val = val_df.iloc[:, 1:-1]
            y_val = val_df.iloc[:, -1]
            
            if self.has_test_labels:
                X_test = test_df.iloc[:, 1:-1]
                y_test = test_df.iloc[:, -1]
                test_sample_ids = test_df.iloc[:, 0]
            else:
                X_test = test_df.iloc[:, 1:]
                y_test = None
                test_sample_ids = test_df.iloc[:, 0]
            
            self.feature_names = X_train.columns.tolist()
            self.target_name = train_df.columns[-1]
            
            print(f"Features: {self.feature_names}")
            print(f"Target: {self.target_name}")
            
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
                
                if y_test is not None:
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
            
            print("=== DATA LOADED SUCCESSFULLY ===")
            
            return {
                'samples_train': len(X_train),
                'samples_val': len(X_val),
                'samples_test': len(X_test),
                'features': X_train.shape[1],
                'task_type': self.task_type,
                'has_test_labels': self.has_test_labels,
                'feature_names': self.feature_names,
                'target_name': self.target_name
            }
        except Exception as e:
            print(f"ERROR in load_data: {str(e)}")
            traceback.print_exc()
            raise
    
    def train_model(self, model_name, **params):
        """Train the selected model"""
        print("=== TRAINING MODEL ===")
        print(f"Model: {model_name}")
        print(f"Params: {params}")
        print(f"Task type: {self.task_type}")
        
        try:
            if self.task_type not in ['classification', 'regression']:
                raise ValueError(f"Task type not set. Current: {self.task_type}")
            
            # FIXED: Use 'class' for classification and 'reg' for regression
            model_key = 'class' if self.task_type == 'classification' else 'reg'
            print(f"Looking for model with key: {model_key}")
            
            model_class = self.models[model_name][model_key]
            print(f"Model class: {model_class}")
            
            self.model = model_class(**params)
            print("Model instantiated")
            
            print("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(data_store['X_train'])
            X_val_scaled = self.scaler.transform(data_store['X_val'])
            print(f"X_train shape: {X_train_scaled.shape}")
            print(f"X_val shape: {X_val_scaled.shape}")
            
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
        """Evaluate the trained model on test data"""
        print("=== EVALUATING MODEL ===")
        try:
            if self.model is None:
                raise ValueError("No model trained yet.")
            
            X_test_scaled = self.scaler.transform(data_store['X_test'])
            y_pred = self.model.predict(X_test_scaled)
            data_store['test_predictions'] = y_pred
            
            if data_store['has_test_labels']:
                y_true = data_store['y_test']
                
                if self.task_type == 'classification':
                    accuracy = accuracy_score(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True)
                    conf_matrix = confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    ax.set_title('Confusion Matrix - Test Set')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    conf_matrix_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close()
                    
                    return {
                        'has_labels': True,
                        'accuracy': accuracy,
                        'classification_report': report,
                        'confusion_matrix': conf_matrix.tolist(),
                        'confusion_matrix_img': conf_matrix_img,
                        'predictions': y_pred.tolist()
                    }
                else:
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_true, y_pred, alpha=0.6)
                    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title('Predictions vs Actual - Test Set')
                    
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
                    ax.set_title('Residual Plot - Test Set')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    residuals_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close()
                    
                    return {
                        'has_labels': True,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'scatter_img': scatter_img,
                        'residuals_img': residuals_img,
                        'predictions': y_pred.tolist()
                    }
            else:
                return {
                    'has_labels': False,
                    'predictions': y_pred.tolist(),
                    'message': 'Test data does not contain target column. Predictions generated successfully.'
                }
        except Exception as e:
            print(f"ERROR in evaluate_model: {str(e)}")
            traceback.print_exc()
            raise
    
    def predict_single(self, X_new):
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        X_scaled = self.scaler.transform(X_new)
        predictions = self.model.predict(X_scaled)
        
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def download_predictions(self):
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        test_sample_ids = data_store['test_sample_ids']
        
        if self.task_type == 'classification':
            predictions_decoded = self.label_encoder.inverse_transform(data_store['test_predictions'].astype(int))
        else:
            predictions_decoded = data_store['test_predictions']
        
        results_df = pd.DataFrame({
            'Sample ID': test_sample_ids,
            'Predictions': predictions_decoded
        })
        
        if data_store['has_test_labels']:
            y_true = data_store['y_test']
            if self.task_type == 'classification':
                y_true_decoded = self.label_encoder.inverse_transform(y_true.astype(int))
            else:
                y_true_decoded = y_true
            results_df['Actual'] = y_true_decoded
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            summary_data = {
                'Metric': ['Model', 'Task Type', 'Number of Predictions', 'Date Generated'],
                'Value': [
                    self.training_history[-1]['model_name'] if self.training_history else 'N/A',
                    self.task_type,
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
            'has_test_labels': trainer.has_test_labels
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