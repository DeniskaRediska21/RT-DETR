import os
from pathlib import Path
from load_to_mlflow import log_model


current_dir = os.getcwd()
dirpath = 'rtdetr-r50-cppe5-finetune'
paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)
log_model(str(Path(current_dir, paths[-1])))
