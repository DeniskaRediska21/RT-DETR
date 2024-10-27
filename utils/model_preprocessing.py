import os
from os import PathLike
from pathlib import Path
import shutil
import yaml
from mlflow import exceptions, transformers, set_tracking_uri, set_experiment
from mlflow.artifacts import download_artifacts
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

def get_path(weights_path: PathLike, model_name: str) -> Path | str:
    client = MlflowClient()
    mlflow_path = 'models:/' + model_name
    local_path = Path(weights_path, model_name)
    if local_path.exists() and any(local_path.iterdir()):
        try:
            current_mlflow_version = client.get_model_version_by_alias(model_name.split('@')[0], model_name.split('@')[1])
        except (exceptions.RestException, exceptions.MlflowException):
            current_mlflow_version = None

        if current_mlflow_version is not None:
            try:
                with Path(local_path, 'registered_model_meta').open('r', encoding="utf-8") as stream:
                    registered_model = yaml.safe_load(stream)
            except FileNotFoundError:
                return mlflow_path
            if current_mlflow_version.version != registered_model['model_version']:
                shutil.rmtree(local_path)
                return mlflow_path
        return local_path
    return mlflow_path


def get_model(mlflow_uri, project_name, model_name):
    """Возвращает модель на Tensorflow.

    Args:
        model_name: Название модели, которая хранится в папке models.

    Returns:
        Модель на Tensorflow / None.

    """
    weights_path = Path(__file__).parent / 'models'
    local_path = Path(weights_path, model_name)
    download_model(mlflow_uri, project_name, model_name)
    model = transformers.load_model(local_path)

    return model


def download_model(mlflow_uri, project_name, model_name):
    tracking_uri = mlflow_uri
    set_tracking_uri(tracking_uri)
    set_experiment(project_name)

    weights_path = Path(__file__).parent / 'models'
    path = get_path(weights_path, model_name)
    local_path = Path(weights_path, model_name)
    local_path.mkdir(parents=True, exist_ok=True)
    if path != local_path:
        try:
            _ = download_artifacts(path, dst_path = str(local_path))
        except MlflowException as exp:
            if tracking_uri is None:
                raise KeyError(f"""Environment variable "MLFLOW_URI" doesn't exist. If you are trying to download model from local MLFlow, set environment variable "MLFLOW_URI" to local MLFlow URI wich has "{config.CLOUDNESS_MODEL['model']}" model""") from exp
            raise exp

    return 0
