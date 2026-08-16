# app.py
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
            'DT': {'class': DecisionTreeClassifier, 'reg': DecisionTreeRegressor},
            'RF': {'class': RandomForestClassifier, 'reg': RandomForestRegressor},
            'SVM': {'class': SVC, 'reg': SVR},
            'DNN': {'class': MLPClassifier, 'reg': MLPRegressor}
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
        # Read all sheets
        xls = pd.ExcelFile(xlsx_file)
        sheet_names = xls.sheet_names
        
        # Check if required sheets exist
        required_sheets = ['Train', 'Val', 'Test']
        for sheet in required_sheets:
            if sheet not in sheet_names:
                raise ValueError(f"Required worksheet '{sheet}' not found. Found: {sheet_names}")
        
        # Load each sheet
        train_df = pd.read_excel(xlsx_file, sheet_name='Train')
        val_df = pd.read_excel(xlsx_file, sheet_name='Val')
        test_df = pd.read_excel(xlsx_file, sheet_name='Test')
        
        # Validate columns
        for df, name in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
            if df.shape[1] < 3:
                raise ValueError(f"{name} sheet must have at least 3 columns (Sample ID, features, target)")
        
        # Check if test has target column
        self.has_test_labels = test_df.shape[1] > 2
        
        # Get column names
        self.sample_id_col = train_df.columns[0]
        
        # Separate features and target
        X_train = train_df.iloc[:, 1:-1]
        y_train = train_df.iloc[:, -1]
        
        X_val = val_df.iloc[:, 1:-1]
        y_val = val_df.iloc[:, -1]
        
        # Test data
        if self.has_test_labels:
            X_test = test_df.iloc[:, 1:-1]
            y_test = test_df.iloc[:, -1]
            test_sample_ids = test_df.iloc[:, 0]
        else:
            X_test = test_df.iloc[:, 1:]
            y_test = None
            test_sample_ids = test_df.iloc[:, 0]
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        self.target_name = train_df.columns[-1]
        
        # Determine task type from training data
        y_train_series = y_train
        if y_train_series.dtype == 'object' or len(y_train_series.unique()) < 10:
            self.task_type = 'classification'
            # Encode labels for classification
            y_train_encoded = self.label_encoder.fit_transform(y_train_series)
            y_train = y_train_encoded
            
            if y_val is not None:
                y_val_encoded = self.label_encoder.transform(y_val)
                y_val = y_val_encoded
            
            if y_test is not None:
                try:
                    y_test_encoded = self.label_encoder.transform(y_test)
                    y_test = y_test_encoded
                except:
                    y_test_encoded = np.array([-1 if label not in self.label_encoder.classes_ else 
                                               self.label_encoder.transform([label])[0] 
                                               for label in y_test])
                    y_test = y_test_encoded
        else:
            self.task_type = 'regression'
        
        # Store data
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
        data_store['train_df_original'] = train_df
        data_store['val_df_original'] = val_df
        data_store['test_df_original'] = test_df
        
        data_store['y_train_original'] = train_df.iloc[:, -1]
        data_store['y_val_original'] = val_df.iloc[:, -1] if y_val is not None else None
        data_store['y_test_original'] = test_df.iloc[:, -1] if self.has_test_labels else None
        
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
    
    def train_model(self, model_name, **params):
        """Train the selected model"""
        if self.task_type not in ['classification', 'regression']:
            raise ValueError("Task type not set. Load data first.")
        
        # Get model class
        model_class = self.models[model_name][self.task_type]
        self.model = model_class(**params)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(data_store['X_train'])
        X_val_scaled = self.scaler.transform(data_store['X_val'])
        
        # Train model
        self.model.fit(X_train_scaled, data_store['y_train'])
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, data_store['y_train'])
        val_score = self.model.score(X_val_scaled, data_store['y_val'])
        
        # Store training info
        self.training_history.append({
            'model_name': model_name,
            'params': params,
            'train_score': train_score,
            'val_score': val_score,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Make predictions on all data
        X_all = pd.concat([data_store['X_train'], data_store['X_val'], data_store['X_test']])
        X_all_scaled = self.scaler.transform(X_all)
        self.all_predictions = self.model.predict(X_all_scaled)
        
        return {
            'train_score': train_score,
            'val_score': val_score,
            'model_name': model_name,
            'params': params,
            'task_type': self.task_type
        }
    
    def evaluate_model(self):
        """Evaluate the trained model on test data"""
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        # Scale test data
        X_test_scaled = self.scaler.transform(data_store['X_test'])
        y_pred = self.model.predict(X_test_scaled)
        
        # Store predictions
        data_store['test_predictions'] = y_pred
        
        # If test has labels, evaluate
        if data_store['has_test_labels']:
            y_true = data_store['y_test']
            
            if self.task_type == 'classification':
                accuracy = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_true, y_pred)
                
                # Create confusion matrix plot
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
            else:  # regression
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                # Create scatter plot
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
                
                # Residuals plot
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
            # No labels in test data - return predictions only
            return {
                'has_labels': False,
                'predictions': y_pred.tolist(),
                'message': 'Test data does not contain target column. Predictions generated successfully.'
            }
    
    def predict_single(self, X_new):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        X_scaled = self.scaler.transform(X_new)
        predictions = self.model.predict(X_scaled)
        
        # Decode if classification
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def download_predictions(self):
        """Download predictions as Excel file"""
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        # Prepare results dataframe with Sample IDs
        test_sample_ids = data_store['test_sample_ids']
        
        # Decode predictions if classification
        if self.task_type == 'classification':
            predictions_decoded = self.label_encoder.inverse_transform(data_store['test_predictions'].astype(int))
        else:
            predictions_decoded = data_store['test_predictions']
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Sample ID': test_sample_ids,
            'Predictions': predictions_decoded
        })
        
        # Add actual values if available
        if data_store['has_test_labels']:
            y_true = data_store['y_test']
            if self.task_type == 'classification':
                y_true_decoded = self.label_encoder.inverse_transform(y_true.astype(int))
            else:
                y_true_decoded = y_true
            results_df['Actual'] = y_true_decoded
        
        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Add summary sheet
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
        
        # Load data from the three worksheets
        data_info = trainer.load_data(filepath)
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'feature_names': trainer.feature_names,
            'target_name': trainer.target_name,
            'has_test_labels': trainer.has_test_labels
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    model_name = data.get('model_name')
    params = data.get('params', {})
    
    if not model_name:
        return jsonify({'error': 'Model name not specified'}), 400
    
    try:
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
        
        results = trainer.train_model(model_name, **params)
        
        return jsonify({
            'success': True,
            'results': results,
            'task_type': trainer.task_type
        })
    
    except Exception as e:
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
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'task_type': trainer.task_type,
        'feature_names': trainer.feature_names,
        'target_name': trainer.target_name,
        'is_trained': trainer.model is not None,
        'training_history': trainer.training_history,
        'has_test_labels': trainer.has_test_labels
    })

@app.route('/clear', methods=['POST'])
def clear_data():
    global data_store
    data_store = {}
    trainer.model = None
    trainer.all_predictions = None
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)