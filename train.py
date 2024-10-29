import torch
import os
import datetime
# from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, pipeline
import mlflow
import numpy as np
from utils import get_model
from transformers import TrainingArguments
from data import LizaDataset, get_transforms
from transformers import Trainer
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.load_to_mlflow import log_model
from utils import convert_bbox_yolo_to_pascal, collate_fn

from config import (
    MLFLOW_URI,
    PROJECT_NAME,
    MODEL_NAME,
    DATASET_PATH,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    DATASET_SIZES,
)


mlflow_uri = MLFLOW_URI
project_name = PROJECT_NAME
model_name = MODEL_NAME
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(project_name)
lengths = DATASET_SIZES 

DEVICE = 'cuda'

pipeline_ = get_model(mlflow_uri, project_name, model_name)
model, image_processor = pipeline_.model, pipeline_.image_processor

image_processor.do_resize = False
image_processor.do_normalize = False

dataset_path = DATASET_PATH
dataset = LizaDataset(dataset_path, image_processor=image_processor, transforms=None)
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, lengths)

model = model.to(DEVICE)

output_dir = os.path.join("rtdetr-r50-cppe5-finetune", datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    max_grad_norm=0.1,
    learning_rate=LEARNING_RATE,
    warmup_steps=300,
    per_device_train_batch_size=BATCH_SIZE,
    dataloader_num_workers=4,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=100,
    logging_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=image_processor,
    data_collator=collate_fn,
)

try:
  trainer.train()
except (Exception, KeyboardInterrupt) as exp:
    log_model(output_dir)
    raise exp
log_model(output_dir)
