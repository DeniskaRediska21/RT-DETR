import torch
import os
import requests
from pathlib import Path
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from transformers import pipeline
import mlflow
from mlflow.models import infer_signature
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../RT-DETR')
from config import (
    MLFLOW_URI,
    PROJECT_NAME,
    MODEL_NAME,
)


def load_to_mlflow(PATH):
    categories = ['person']
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    DEVICE = 'cpu'
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = RTDetrImageProcessor.from_pretrained(PATH, local_files_only=True)
    model = RTDetrForObjectDetection.from_pretrained(PATH,
                                                     local_files_only=True,
                                                     id2label=id2label,
                                                     label2id=label2id,
                                                     ignore_mismatched_sizes=True)

    inputs = image_processor(images=image, return_tensors="pt")

    model = model.to(DEVICE)
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    mlflow_uri = MLFLOW_URI
    project_name = PROJECT_NAME
    model_name = MODEL_NAME.split('@')[0]

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(project_name)

    signature_DETR = infer_signature(model_input={
        'images': np.array(image),
    },
                                     model_output={
                                         k: np.array(v) for k, v in outputs.items() if not isinstance(v, list)
                                     })

    pipe = pipeline(model=model.cpu(), image_processor=image_processor, device='cpu', task='object-detection')

    mlflow.transformers.log_model(
        pipe,
        model_name,
        registered_model_name=model_name,
        signature=signature_DETR,
    )

    # results = image_processor.post_process_object_detection(outputs,
    #                                                         target_sizes=torch.tensor([image.size[::-1]]),
    #                                                         threshold=0.3)

    # for result in results:
    #     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
    #         score, label = score.item(), label_id.item()
    #         box = [round(i, 2) for i in box.tolist()]
    #         print(f"{model.config.id2label[label]}: {score:.2f} {box}")


def log_model(savedir):
    checkpoint_dirs = os.listdir(savedir)
    last_checkpoint = np.argmax([int(''.join([ch for ch in filename if 47<ord(ch)<58])) for filename in checkpoint_dirs])
    last_checkpoint_dir = checkpoint_dirs[last_checkpoint]
    PATH = os.path.join(savedir, last_checkpoint_dir)
    load_to_mlflow(PATH)


if __name__ == '__main__':
    # PATH = Path(os.sep, 'home', 'user', 'LIZA', 'RT-DETR', 'weights', 'RT_DETR_HF')
    # load_to_mlflow(PATH)
    PATH = Path(os.sep,
                'home',
                'user',
                'LIZA',
                'RT-DETR',
                'rtdetr-r50-cppe5-finetune',
                'October_28_2024_10_11_54',
        )
    log_model(PATH)

