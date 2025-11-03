"""
Data Intelligence Platform - Main Server
Advanced feature optimization system powered by evolutionary algorithms
"""
import os

# Configure single-threaded execution to prevent multiprocessing conflicts
os.environ['JOBLIB_MULTIPROCESSING'] = '0'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from engines import EvolutionOptimizer, BaselineSelector

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'storage'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['SECRET_KEY'] = 'data-intelligence-platform-2025'

# Create storage directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


def validate_file_extension(filename):
    """
    Validate uploaded file extension
    Args:
        filename: Name of the uploaded file
    Returns:
        Boolean indicating if file extension is allowed
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """
    Landing page route - Introduction to the platform
    """
    return render_template('home.html')


@app.route('/demo')
def demo():
    """
    Demo page route - Interactive feature optimization tool
    """
    return render_template('demo.html')


@app.route('/api/preview', methods=['POST'])
def preview_dataset():
    """
    API endpoint to preview uploaded dataset structure
    Returns column names and basic statistics
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not validate_file_extension(file.filename):
            return jsonify({'error': 'Only CSV and Excel files are supported'}), 400
        
        # Read dataset to extract metadata (supports CSV and Excel)
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(file)
        else:  # xlsx or xls
            df = pd.read_excel(file, engine='openpyxl' if file_ext == 'xlsx' else None)
        columns = df.columns.tolist()
        
        return jsonify({
            'columns': columns, 
            'total_rows': int(df.shape[0]), 
            'total_columns': int(df.shape[1])
        })
    
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500


@app.route('/api/optimize', methods=['POST'])
def optimize_features():
    """
    API endpoint to run feature optimization
    Processes uploaded dataset and returns optimization results
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not validate_file_extension(file.filename):
            return jsonify({'error': 'Only CSV and Excel files are supported'}), 400
        
        # Save file temporarily for processing
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load dataset (supports CSV and Excel)
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(filepath)
        else:  # xlsx or xls
            df = pd.read_excel(filepath, engine='openpyxl' if file_ext == 'xlsx' else None)
        
        # Extract target column from request
        target_column = request.form.get('target_column')
        if not target_column or target_column not in df.columns:
            target_column = df.columns[-1]  # Default to last column
        
        # Prepare feature matrix and target vector
        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = df[target_column]
        
        # Handle missing values using mean imputation
        X = X.fillna(X.mean())
        
        # Convert categorical target to numeric codes if needed
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
        
        # Extract optimization parameters from request
        population_size = int(request.form.get('population_size', 50))
        generations = int(request.form.get('generations', 30))
        mutation_rate = float(request.form.get('mutation_rate', 0.1))
        
        # Execute evolutionary optimization
        optimizer = EvolutionOptimizer(
            X.values, y.values,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            callback=None
        )
        optimized_features, optimization_score, evolution_history = optimizer.evolve()
        
        # Compare with baseline methods
        baseline = BaselineSelector(X.values, y.values)
        comparison_results = baseline.compare_all(optimized_features, optimization_score)
        
        # Prepare comprehensive results
        results = {
            'success': True,
            'dataset_info': {
                'total_rows': int(df.shape[0]),
                'total_features': int(X.shape[1]),
                'selected_features': len(optimized_features)
            },
            'evolutionary_optimization': {
                'accuracy': float(optimization_score),
                'feature_count': len(optimized_features),
                'selected_indices': optimized_features,
                'feature_names': [X.columns[i] for i in optimized_features],
                'evolution_history': evolution_history
            },
            'baseline_comparison': {
                'methods': ['Evolutionary', 'Statistical Test', 'Information Theory', 'Recursive', 'Tree-Based'],
                'accuracy_scores': [
                    float(comparison_results['evolutionary']['accuracy']),
                    float(comparison_results['statistical']['accuracy']),
                    float(comparison_results['information']['accuracy']),
                    float(comparison_results['recursive']['accuracy']),
                    float(comparison_results['tree_based']['accuracy'])
                ],
                'feature_counts': [
                    comparison_results['evolutionary']['feature_count'],
                    comparison_results['statistical']['feature_count'],
                    comparison_results['information']['feature_count'],
                    comparison_results['recursive']['feature_count'],
                    comparison_results['tree_based']['feature_count']
                ]
            }
        }
        
        # Clean up temporary file
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        # Clean up file on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

