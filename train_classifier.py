from tqdm import tqdm
from utils.model_preprocessing import get_transformers_model
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.models as models
import torch.nn as nn
import logging
from mlflow.utils import logging_utils
from torcheval.metrics import MulticlassAccuracy
import mlflow
from torchvision.ops import Conv2dNormActivation
from torch.nn import SiLU
from data.dataloader import get_train_val_dataloader
from utils.classifier_transforms import get_resize_transform, get_train_transform, ge

from config import (
    CLASS_DATASET_PATH,
    CLASS_DEVICE,
    CLASS_LEARNING_RATE,
    CLASS_NUM_EPOCHS,
    CLASS_BATCH_SIZE,
    CLASS_NUM_WORKERS,
    CLASS_NUM_CLASSES,
    CLASS_INFERENCE_SIZE,
    CLASS_TRAIN_SIZE,
    CLASS_LOG_STEP,
    PROJECT_NAME,
    MLFLOW_URI,
    CLASS_WEIGHT_DECAY,
    CLASS_TYPE,
    PROJECT_NAME,
    MODEL_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%d.%m %H:%M:%S'
)
log = logging.getLogger()
logging_utils.disable_logging()  # MLflow throws some excessive logging warnings


def build_model(pretrained=True, fine_tune=True, num_classes=10, n_bands=3, type='effnet_b0'):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    match type:
        case 'effnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
        case 'effnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
        case 'effnet_b7':
            model = models.efficientnet_b7(pretrained=pretrained)
        case 'effnet_s':
            model = models.efficientnet_v2_s(pretrained=pretrained)
        case 'effnet_m':
            model = models.efficientnet_v2_m(pretrained=pretrained)
        case 'effnet_l':
            model = models.efficientnet_v2_l(pretrained=pretrained)
        case _:
            NotImplementedError(f'Model {type} not implemented')

    print(f'[INFO]: Using model {type}')
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    in_features = next(model.classifier[1].parameters()).size()[1]
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    if n_bands != 3:
        print(f'Constructing model for {n_bands}-band images')
        out_channels = len(list(model.features[0][1].named_parameters())[0][1])
        model.features[0] = Conv2dNormActivation(
             in_channels=n_bands,
             out_channels=out_channels,
             kernel_size=(3, 3),
             stride=(2, 2),
             padding=(1, 1),
             bias=False,
             activation_layer=SiLU
             )
    return model


def train(classification_model, oprimizer, criterion, dataloader: DataLoader, epoch: int, verbose: bool = True, resize_transform=None):
    progress_bar = tqdm(dataloader, total=len(dataloader), desc=' ')
    metric = MulticlassAccuracy()
    classification_model = classification_model.to(CLASS_DEVICE)

    for i, data in enumerate(progress_bar):
        images, targets = data
        if resize_transform is not None:
            images = [resize_transform(image) for image in images]

        images = torch.stack([image.to(CLASS_DEVICE) for image in images])
        targets = torch.stack(targets).to(CLASS_DEVICE)

        optimizer.zero_grad()

        outputs = classification_model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        metric.update(outputs, targets)

        accuracy = metric.compute()
        progress_bar.set_description(f'Training: loss= {loss:.4f}, accuracy= {accuracy:.4f}')
        step = epoch * len(dataloader) + i
        if step % CLASS_LOG_STEP == 0:
            mlflow.log_metric('train/loss', loss, step=step)
            mlflow.log_metric('train/accuracy', accuracy, step=step)

    mlflow.pytorch.log_model(classification_model, 'last')


def validate(classification_model, dataloader: DataLoader, epoch: int, verbose: bool = True, resize_transform=None):
    progress_bar = tqdm(dataloader, total=len(dataloader), desc=' ')
    metric = MulticlassAccuracy()
    classification_model = classification_model.to(CLASS_DEVICE)
    for i, data in enumerate(progress_bar):
        images, targets = data
        if resize_transform is not None:
            images = [resize_transform(image) for image in images]

        images = torch.stack([image.to(CLASS_DEVICE) for image in images])
        targets = torch.stack(targets).to(CLASS_DEVICE)

        outputs = classification_model(images.to(CLASS_DEVICE))
        loss = criterion(outputs, targets)
        metric.update(outputs, targets)
        accuracy = metric.compute()
        progress_bar.set_description(f'Validation: loss= {loss:.4f}, accuracy= {accuracy:.4f}')

        step = epoch * len(dataloader) + i
        if step % CLASS_LOG_STEP == 0:
            mlflow.log_metric('val/loss', loss, step=step)
            mlflow.log_metric('val/accuracy', accuracy, step=step)


if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(PROJECT_NAME)

    dataset_dir = CLASS_DATASET_PATH
    image_dir = 'images'
    annotation_dir = 'annotations'
    annotation_bboxes_dir = 'annotations_bboxes'

    project_name = PROJECT_NAME
    model_name = MODEL_NAME
    mlflow_uri = MLFLOW_URI

    pipeline_ = get_transformers_model(mlflow_uri, project_name, model_name)
    model, image_processor = pipeline_.model, pipeline_.image_processor

    image_processor.do_resize = False
    image_processor.do_normalize = False
    image_processor.do_pad = False

    train_dataloader, val_dataloader = get_train_val_dataloader(
                         dataset_dir=CLASS_DATASET_PATH,
                         batch_size=CLASS_BATCH_SIZE,
                         num_workers=CLASS_NUM_WORKERS,
                         image_processor = image_processor,
                         shuffle=True,
                         transforms=get_train_transform(CLASS_INFERENCE_SIZE, True),
                         train_size=CLASS_TRAIN_SIZE
     )

    log.info('Training...')

    model = build_model(
        pretrained=True,
        fine_tune=True,
        num_classes=CLASS_NUM_CLASSES,
        n_bands=3,
        type=CLASS_TYPE,
    )

    optimizer = Adam(model.parameters(), lr=CLASS_LEARNING_RATE, weight_decay=CLASS_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    resize_transform = get_resize_transform(INFERENCE_SIZE=CLASS_INFERENCE_SIZE)

    exp = None
    try:
        for epoch in range(CLASS_NUM_EPOCHS):
            log.info(f'Epoch {epoch + 1}:')

            train(classification_model=model, oprimizer=optimizer, criterion=criterion, dataloader=train_dataloader, epoch=epoch, resize_transform=resize_transform)
            validate(classification_model=model, dataloader=val_dataloader, epoch=epoch, resize_transform=resize_transform)
    except Exception as exeption:
        exp = exeption

    hyperparameters = {
        'epochs': epoch,
        'batch_size': CLASS_BATCH_SIZE,
        'inference_size': CLASS_INFERENCE_SIZE,
        'learning_rate': CLASS_LEARNING_RATE,
        'optimizer': 'Adam',
        'weight_decay': CLASS_WEIGHT_DECAY,
        'train_size': CLASS_TRAIN_SIZE,
        'num_classes': CLASS_NUM_CLASSES,
        'dataset': CLASS_DATASET_PATH,
        'exeption': repr(exp),
        'model_type': CLASS_TYPE,
    }
    mlflow.log_params(hyperparameters)

    if exp:
        raise exp
