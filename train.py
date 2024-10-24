import torch
import requests
from pathlib import Path
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from transformers import pipeline
import mlflow
import numpy as np
from model_preprocessing import get_model

from data import LizaDataset


mlflow_uri = 'http://localhost:5000'
project_name = 'LIZA'
model_name = 'LIZA-detector@base'
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(project_name)

dataset = LizaDataset(os.path.join('..', 'Dataset'), None)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
image = Image.open(requests.get(url, stream=True).raw)

DEVICE = 'cuda'

pipline = get_model(mlflow_uri, project_name, model_name)
model, image_processor = pipline.model, pipline.image_processor

inputs = image_processor(images=image, return_tensors="pt")

model = model.to(DEVICE)
inputs = inputs.to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)


results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
