import torch
import requests
from pathlib import Path
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from transformers import pipeline
import mlflow
from mlflow.models import infer_signature
import numpy as np


DEVICE = 'cpu'
url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
image = Image.open(requests.get(url, stream=True).raw)

PATH = Path('weights', 'RT_DETR_HF')
image_processor = RTDetrImageProcessor.from_pretrained(PATH, local_files_only=True)
model = RTDetrForObjectDetection.from_pretrained(PATH, local_files_only=True)

inputs = image_processor(images=image, return_tensors="pt")

model = model.to(DEVICE)
inputs = inputs.to(DEVICE)
with torch.no_grad():
    outputs = model(**inputs)

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('LIZA')

signature_DETR = infer_signature(
        model_input={
                    'images' : np.array(image),
                    }, 
        model_output={k: np.array(v) for k, v in outputs.items() if not isinstance(v, list)}
)

model_name = 'LIZA-detector'

pipe = pipeline(model = model.cpu(), image_processor=image_processor,device='cpu', task='object-detection')

mlflow.transformers.log_model(
    pipe, 
    model_name, 
    registered_model_name=model_name, 
    signature = signature_DETR,
)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
