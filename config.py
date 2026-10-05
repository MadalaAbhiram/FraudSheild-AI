import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY          = os.environ.get('SECRET_KEY') or 'fraud-detection-secret-key-2024'
    UPLOAD_FOLDER       = os.path.join(BASE_DIR, 'uploads')
    MODEL_FOLDER        = os.path.join(BASE_DIR, 'models', 'saved_models')
    MAX_CONTENT_LENGTH  = 500 * 1024 * 1024   # 500 MB
    ALLOWED_EXTENSIONS  = {'csv'}
    DEBUG               = True

    # EDA runs on a stratified sample for speed.
    # Full data is always used for training and prediction.
    EDA_SAMPLE_ROWS     = 50_000
