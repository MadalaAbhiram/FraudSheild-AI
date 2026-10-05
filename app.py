"""
app.py
======
Main Flask application for the Credit Card Fraud Detection System.
"""

import os
import json
import traceback

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, send_from_directory, flash,
)
from werkzeug.utils import secure_filename

from config import Config
from utils.helpers import allowed_file, get_file_size, cleanup_old_files, format_number
from eda.analysis import EDAAnalyzer
from preprocessing.preprocess import FraudPreprocessor
from models.train_model import FraudModelTrainer, PREPROCESSOR_PATH
from models.predict import FraudPredictor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)

# Custom Jinja2 filter
app.jinja_env.filters['format_number'] = format_number

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

_EDA_CACHE = {}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _upload_path(filename: str) -> str:
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)


def _get_eda_results(filepath: str) -> tuple[pd.DataFrame, dict, list[dict]]:
    sample_rows = app.config['EDA_SAMPLE_ROWS']
    cache_key = (filepath, os.path.getmtime(filepath), os.path.getsize(filepath), sample_rows)
    cached = _EDA_CACHE.get(cache_key)
    if cached:
        return cached

    proc = FraudPreprocessor()
    df, info = proc.load_data(filepath, max_rows=sample_rows)
    analyzer = EDAAnalyzer(df)
    steps = analyzer.run_all()

    _EDA_CACHE.clear()
    _EDA_CACHE[cache_key] = (df, info, steps)
    return df, info, steps


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---- Upload ---------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    train_file = request.files.get('train_file')
    test_file  = request.files.get('test_file')

    # At least one file must be provided
    if (not train_file or train_file.filename == '') and \
       (not test_file  or test_file.filename  == ''):
        flash('Please select at least one CSV file.', 'danger')
        return redirect(url_for('index'))

    saved = {}

    # Save train file
    if train_file and train_file.filename != '':
        if not allowed_file(train_file.filename):
            flash('Train file must be a CSV.', 'danger')
            return redirect(url_for('index'))
        train_name = secure_filename(train_file.filename)
        train_path = _upload_path(train_name)
        train_file.save(train_path)
        saved['train'] = train_name
        session['train_file'] = train_name
        session['current_file'] = train_name   # default EDA target

    # Save test file
    if test_file and test_file.filename != '':
        if not allowed_file(test_file.filename):
            flash('Test file must be a CSV.', 'danger')
            return redirect(url_for('index'))
        test_name = secure_filename(test_file.filename)
        test_path = _upload_path(test_name)
        test_file.save(test_path)
        saved['test'] = test_name
        session['test_file'] = test_name

    cleanup_old_files(app.config['UPLOAD_FOLDER'], max_files=10)

    msg_parts = []
    if 'train' in saved:
        msg_parts.append(f'Train: {saved["train"]} ({get_file_size(_upload_path(saved["train"]))})')
    if 'test' in saved:
        msg_parts.append(f'Test: {saved["test"]} ({get_file_size(_upload_path(saved["test"]))})')
    flash('Uploaded — ' + ' | '.join(msg_parts), 'success')

    # Redirect to EDA using the train file (preferred) or whichever was uploaded
    eda_target = saved.get('train') or saved.get('test')
    return redirect(url_for('eda', filename=eda_target))


# ---- EDA ------------------------------------------------------------------
@app.route('/eda/<filename>')
def eda(filename: str):
    filepath = _upload_path(filename)
    if not os.path.exists(filepath):
        flash('File not found. Please upload again.', 'danger')
        return redirect(url_for('index'))

    session['current_file'] = filename

    try:
        _, info, steps = _get_eda_results(filepath)
    except Exception as exc:
        flash(f'EDA error: {exc}', 'danger')
        return redirect(url_for('index'))

    return render_template(
        'eda.html',
        filename=filename,
        steps=steps,
        info=info,
        file_size=get_file_size(filepath),
    )


# ---- EDA single-step API --------------------------------------------------
@app.route('/api/eda-step/<int:step>/<filename>')
def api_eda_step(step: int, filename: str):
    filepath = _upload_path(filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        _, _, steps = _get_eda_results(filepath)
        if step < 1 or step > len(steps):
            return jsonify({'error': 'Invalid step number'}), 400
        result = steps[step - 1]
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc), 'traceback': traceback.format_exc()}), 500


# ---- Model Training page --------------------------------------------------
@app.route('/model/<filename>')
def model_page(filename: str):
    filepath = _upload_path(filename)
    if not os.path.exists(filepath):
        flash('File not found.', 'danger')
        return redirect(url_for('index'))
    session['current_file'] = filename
    return render_template('model.html', filename=filename)


# ---- Train ----------------------------------------------------------------
@app.route('/train/<filename>', methods=['POST'])
def train(filename: str):
    filepath = _upload_path(filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    use_smote = request.json.get('use_smote', False) if request.is_json else False

    try:
        proc = FraudPreprocessor()
        pipeline = proc.full_pipeline(filepath)

        if 'X_train' not in pipeline:
            return jsonify({'error': 'Dataset has no target column (is_fraud). Cannot train.'}), 400

        # Save preprocessor for later prediction
        joblib.dump(proc, PREPROCESSOR_PATH)

        trainer = FraudModelTrainer()
        results = trainer.train_all(
            pipeline['X_train'], pipeline['X_test'],
            pipeline['y_train'], pipeline['y_test'],
            pipeline['feature_names'],
            use_smote=use_smote,
        )

        # Attach feature importance plot data
        best = results['best_model']
        fi = results['results'].get(best, {}).get('feature_importance', [])

        return jsonify({
            'success': True,
            'results': results['results'],
            'unsupervised_results': results.get('unsupervised_results', {}),
            'pca_info': pipeline.get('pca_info', {}),
            'best_model': best,
            'feature_importance': fi,
            'feature_names': pipeline['feature_names'],
        })
    except Exception as exc:
        return jsonify({'error': str(exc), 'traceback': traceback.format_exc()}), 500


# ---- Predict page ---------------------------------------------------------
@app.route('/predict/<filename>')
def predict_page(filename: str):
    filepath = _upload_path(filename)
    if not os.path.exists(filepath):
        flash('File not found.', 'danger')
        return redirect(url_for('index'))
    session['current_file'] = filename

    from models.train_model import BEST_MODEL_PATH
    model_ready = os.path.exists(BEST_MODEL_PATH)

    # Offer test file as prediction target if it was uploaded
    test_file  = session.get('test_file', '')
    train_file = session.get('train_file', filename)

    return render_template(
        'predict.html',
        filename=filename,
        model_ready=model_ready,
        test_file=test_file,
        train_file=train_file,
    )


# ---- Run batch prediction -------------------------------------------------
@app.route('/predict/<filename>', methods=['POST'])
def run_prediction(filename: str):
    filepath = _upload_path(filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        predictor = FraudPredictor()
        if not predictor.load_model():
            return jsonify({'error': 'No trained model found. Train a model first.'}), 400

        results_df = predictor.predict_from_file(filepath)

        # Save results CSV
        results_filename = f'results_{filename}'
        results_path = _upload_path(results_filename)
        results_df.to_csv(results_path, index=False)

        # Summary stats
        total = len(results_df)
        fraud_count = int(results_df['prediction'].sum())
        legit_count = total - fraud_count
        fraud_pct = round(fraud_count / total * 100, 2) if total > 0 else 0

        amt_at_risk = 0.0
        if 'amt' in results_df.columns:
            amt_at_risk = float(results_df.loc[results_df['prediction'] == 1, 'amt'].sum())

        # Preview (first 100 rows)
        preview = results_df.head(100).fillna('').to_dict(orient='records')

        return jsonify({
            'success': True,
            'total': total,
            'fraud_count': fraud_count,
            'legit_count': legit_count,
            'fraud_pct': fraud_pct,
            'amt_at_risk': round(amt_at_risk, 2),
            'results_filename': results_filename,
            'preview': preview,
            'model_name': predictor.model_name,
        })
    except Exception as exc:
        return jsonify({'error': str(exc), 'traceback': traceback.format_exc()}), 500


# ---- Manual single-transaction prediction ---------------------------------
@app.route('/api/predict-single', methods=['POST'])
def predict_single():
    data = request.get_json(force=True)
    try:
        predictor = FraudPredictor()
        if not predictor.load_model():
            return jsonify({'error': 'No trained model found.'}), 400
        result = predictor.predict_single(data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ---- Results page ---------------------------------------------------------
@app.route('/results/<filename>')
def results(filename: str):
    filepath = _upload_path(filename)
    if not os.path.exists(filepath):
        flash('Results file not found.', 'danger')
        return redirect(url_for('index'))

    df = pd.read_csv(filepath)
    total = len(df)
    fraud_count = int(df['prediction'].sum()) if 'prediction' in df.columns else 0
    legit_count = total - fraud_count
    fraud_pct = round(fraud_count / total * 100, 2) if total > 0 else 0
    amt_at_risk = 0.0
    if 'amt' in df.columns and 'prediction' in df.columns:
        amt_at_risk = float(df.loc[df['prediction'] == 1, 'amt'].sum())

    preview = df.head(100).fillna('').to_dict(orient='records')
    columns = list(df.columns)

    return render_template(
        'results.html',
        filename=filename,
        total=total,
        fraud_count=fraud_count,
        legit_count=legit_count,
        fraud_pct=fraud_pct,
        amt_at_risk=amt_at_risk,
        preview=preview,
        columns=columns,
    )


# ---- Download -------------------------------------------------------------
@app.route('/download/<filename>')
def download(filename: str):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        as_attachment=True,
    )


# ---- Health check ---------------------------------------------------------
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'app': 'FraudShield AI'})

# ---- Algorithms / ML Foundation page -------------------------------------
@app.route('/algorithms')
def algorithms():
    return render_template('algorithms.html')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)
