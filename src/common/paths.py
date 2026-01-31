from pathlib import Path


__current_dir = Path(__file__).parent
PROJECT_DIR = __current_dir.parent.parent
DATA_DIR = PROJECT_DIR / "data"
