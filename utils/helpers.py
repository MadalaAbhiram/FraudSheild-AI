import os
import glob
from config import Config


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension (CSV only)."""
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def format_number(n) -> str:
    """Format a large number with commas for readability."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)


def get_file_size(filepath: str) -> str:
    """Return a human-readable file size string for the given file path."""
    try:
        size_bytes = os.path.getsize(filepath)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        else:
            return f"{size_bytes / (1024 ** 3):.2f} GB"
    except (OSError, FileNotFoundError):
        return "Unknown"


def cleanup_old_files(folder: str, max_files: int = 5) -> None:
    """
    Keep the uploads folder clean by removing the oldest CSV files when
    the number of files exceeds *max_files*.
    """
    pattern = os.path.join(folder, '*.csv')
    csv_files = sorted(glob.glob(pattern), key=os.path.getmtime)
    while len(csv_files) > max_files:
        oldest = csv_files.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass
