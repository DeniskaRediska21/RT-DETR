import torch
from torchvision.ops import box_convert, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import numpy as np
from utils import get_model
from data import LizaDataset
from data.transforms import get_testtime_transforms
from utils.NMS import remove_overlaping
from pprint import pprint
from config import (
    MLFLOW_URI,
    PROJECT_NAME,
    MODEL_NAME_VAL,
    DATASET_PATH,
    VALIDATION_DATASET_PATH,
    INFERENCE_SIZE,
    OVERLAP,
    NMS_IOU_TRESHOLD,
    RATIO_TRESH,
    INFERENCE_BATCH_SIZE,
    VERBOSE,
    TRESHOLD,
    TESTTIME_IOU_TRESH,
    DO_TESTTIME_AUGMENT,
)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

COLOR_VAL = [0., 1., 0.]

metric = MeanAveragePrecision(box_format='xywh')


def plot_results(pil_img, postprocessed_outputs, additions, width, height, target):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    # for output, addition in zip(postprocessed_outputs, additions):
    scores = postprocessed_outputs['scores'].to('cpu')
    labels = postprocessed_outputs['labels'].to('cpu')
    boxes = postprocessed_outputs['boxes'].to('cpu')

    boxes_val = box_convert(target * torch.tensor([width, height, width, height]), 'xywh', 'xyxy').to('cpu')

    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        w, h = xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[int(label)]}: {score:0.4f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    for xmin, ymin, xmax, ymax in boxes_val:
        w, h = xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle((xmin - w/2, ymin - h/2), w, h,
                                   fill=False, color=COLOR_VAL, linewidth=3))
        
    plt.axis('off')
    plt.show()


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == '__main__':
    mlflow_uri = MLFLOW_URI
    project_name = PROJECT_NAME

    DEVICE = 'cuda'

    pipeline_ = get_model(mlflow_uri, project_name, MODEL_NAME_VAL)

    model, image_processor = pipeline_.model, pipeline_.image_processor

    image_processor.do_resize = False
    image_processor.do_normalize = False
    image_processor.do_pad = False

    dataset_path = VALIDATION_DATASET_PATH
    dataset = LizaDataset(dataset_path, image_processor=image_processor, transforms=None)

    model = model.to(DEVICE)

    testtime_transform = get_testtime_transforms()

    for n_image in range(len(dataset)):
        inputs = dataset.__getitem__(n_image)
        image = inputs['pixel_values']
        _, height, width = image.size()
        step_size = int(INFERENCE_SIZE * (1 - OVERLAP))

        # Sliding window
        h_range = np.array(range(0, height-INFERENCE_SIZE+step_size, step_size))
        w_range = np.array(range(0, width-INFERENCE_SIZE+step_size, step_size))

        h_range[-1] = height - INFERENCE_SIZE if h_range[-1] + INFERENCE_SIZE > height else h_range[-1]
        w_range[-1] = width - INFERENCE_SIZE if w_range[-1] + INFERENCE_SIZE > width else w_range[-1]

        subs = [image[:, ymin: ymin + INFERENCE_SIZE, xmin: xmin + INFERENCE_SIZE] for ymin in h_range for xmin in w_range]

        additions = torch.tensor([(ymin, xmin) for ymin in h_range for xmin in w_range])

        subs = torch.stack(subs)
        batch_size = INFERENCE_BATCH_SIZE
        batches = torch.split(subs, batch_size)
        batched_additions = torch.split(additions, batch_size)

        outputs_all = AttrDict()
        postprocessed_outputs_squeezed = AttrDict()

        for batch, addition in zip(batches, batched_additions):
            with torch.no_grad():
                batch = batch.to(DEVICE)

                outputs = model(batch)

                postprocessed_outputs = image_processor.post_process_object_detection(
                    outputs,
                    target_sizes=[(INFERENCE_SIZE, INFERENCE_SIZE)],
                    threshold=TRESHOLD,
                )
                postprocessed_outputs = postprocessed_outputs[0]

                if DO_TESTTIME_AUGMENT:
                    outputs_testtime = model(testtime_transform(batch))

                    postprocessed_outputs_testtime = image_processor.post_process_object_detection(
                        outputs_testtime,
                        target_sizes=[(INFERENCE_SIZE, INFERENCE_SIZE)],
                        threshold=TRESHOLD,
                    )

                    postprocessed_outputs_testtime = postprocessed_outputs_testtime[0]

                    if len(postprocessed_outputs_testtime['boxes']):
                        xmin = postprocessed_outputs_testtime['boxes'][:, 0]
                        ymin = postprocessed_outputs_testtime['boxes'][:, 1]
                        xmax = postprocessed_outputs_testtime['boxes'][:, 2]
                        ymax = postprocessed_outputs_testtime['boxes'][:, 3]
                    
                        postprocessed_outputs_testtime['boxes'] = torch.stack(
                           [
                               INFERENCE_SIZE - xmax,
                               INFERENCE_SIZE - ymax,
                               INFERENCE_SIZE - xmin,
                               INFERENCE_SIZE - ymin,
                           ]
                        ).T

                    iou = box_iou(postprocessed_outputs['boxes'], postprocessed_outputs_testtime['boxes'])

                    keep, _ = torch.where(iou > TESTTIME_IOU_TRESH)

                    # postprocessed_outputs = postprocessed_outputs_testtime

                    for key, value in postprocessed_outputs.items():
                        postprocessed_outputs[key] = value[keep]

                    # for key in postprocessed_outputs.keys():
                    #     postprocessed_outputs[key] = torch.cat(
                    #          [
                    #              postprocessed_outputs[key],
                    #              postprocessed_outputs_testtime[key],
                    #          ],
                    #          dim = 0,
                    #     )

                addition = addition[0]
                if len(postprocessed_outputs['boxes']):
                    postprocessed_outputs['boxes'][:,0] += addition[1]
                    postprocessed_outputs['boxes'][:,1] += addition[0]
                    postprocessed_outputs['boxes'][:,2] += addition[1]
                    postprocessed_outputs['boxes'][:,3] += addition[0]
            
                for key, value in postprocessed_outputs.items():
                    if key not in postprocessed_outputs_squeezed:
                        postprocessed_outputs_squeezed[key] = []
                    [postprocessed_outputs_squeezed[key].append(val) for val in value]

        for key, value in postprocessed_outputs_squeezed.items():
            if len(value) > 0:
                postprocessed_outputs_squeezed[key] = torch.stack(value)
            else:
                postprocessed_outputs_squeezed[key] = torch.tensor([]).to(DEVICE)

        postprocessed_outputs_squeezed = remove_overlaping(postprocessed_outputs_squeezed, NMS_IOU_TRESHOLD, ratio_tresh=RATIO_TRESH)

        outputs_for_comparison = AttrDict()
        for key, value in postprocessed_outputs_squeezed.items():
            if key == 'boxes' and len(value) > 0:
                # value = value / torch.tensor([width, height, width, height]).to(DEVICE)
                value = box_convert(value, 'xyxy', 'cxcywh')
            if key == 'score' and len(value) > 0:
                value = value * 15
            outputs_for_comparison[key] = value
            # TODO: if saving convert to cxcywh

        metric.update(
            [outputs_for_comparison],
            [dict(
                 boxes=inputs['labels']['boxes'].to(DEVICE),
                 labels=inputs['labels']['class_labels'].to(DEVICE)
            )]
        )
        pprint(metric.compute())

        # pprint(outputs_for_comparison)
        # pprint(dict(
        #      boxes=inputs['labels']['boxes'].to(DEVICE) * torch.tensor([width, height, width, height]).to(DEVICE),
        #      labels=inputs['labels']['class_labels'].to(DEVICE)
        # ))
        if VERBOSE:
            plot_results(image.numpy().transpose((1, 2, 0)), postprocessed_outputs_squeezed, additions, width, height, inputs['labels']['boxes'])
