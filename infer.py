import os
import time
import copy
from torchmetrics.functional import fbeta_score
import torch
# import torch_tensorrt
from torchvision.ops import box_convert, box_iou
import torch.nn.utils.prune as prune
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_transformers_model, get_pytorch_model
from data import LizaDataset
from data.transforms import get_testtime_transforms
from utils.NMS import remove_overlaping, reclassify
from pprint import pprint
from config import (
    MLFLOW_URI,
    DO_CLASSIFY,
    PROJECT_NAME,
    MODEL_NAME_VAL,
    CLASSIFIER_NAME,
    DATASET_PATH,
    VALIDATION_DATASET_PATH,
    INFERENCE_SIZE,
    OVERLAP,
    NMS_IOU_TRESHOLD,
    RATIO_TRESH,
    INFERENCE_BATCH_SIZES,
    VERBOSE,
    TRESHOLD,
    TESTTIME_IOU_TRESH,
    DO_TESTTIME_AUGMENT,
    TESTTIME_VARIANT,
    DO_NMS,
    DEVICE,
    DO_COMPILE,
)

class UpgreatMetric():
    def __init__(self, iou_min=0.5, iou_max=1., iou_step=0.05, beta=1., gamma=0.15, tn=2., ignore=None):
        self.mean_time_score = []
        self.mean_score = 0
        self.beta = beta
        self.gamma = gamma
        self.tn = tn
        self.iou_tresholds = list(np.arange(iou_min, iou_max, iou_step, dtype=np.float32))
        self.ignore = ignore
        self.TP = []
        self.FN = []
        self.FP = []

    def update(self, outputs, targets, elapsed_time):
        if len(outputs) > 0:
            keep = torch.tensor([index for index, label in enumerate(outputs['labels']) if label not in self.ignore])
            for key, value in outputs.items():
                if len(keep) > 0:
                    outputs[key] = value[keep]
                else:
                    outputs[key] = torch.tensor([])
            n_predictions = len(outputs['boxes'])
        else:
            n_predictions = 0

        n_targets = len(targets['boxes'])

        if len(outputs) and len(outputs['boxes']) > 0 and len(targets['boxes']) > 0:
            iou = box_iou(
                box_convert(outputs['boxes'], 'xywh', 'xyxy'),
                box_convert(targets['boxes'], 'xywh', 'xyxy')
            ) # NxM N-outputs, M-targets
            F_beta = []

            for current_treshold in self.iou_tresholds:
                iou_copy = copy.deepcopy(iou)
                # rows, columns = iou_copy.shape
                TP = 0
                for _ in range(n_targets):
                    max_ = torch.max(iou_copy)
                    if max_ >= current_treshold:
                        max_r, max_c = np.unravel_index(torch.argmax(iou_copy).cpu(), iou_copy.shape)
                        TP += 1
                        # rows -=1
                        # columns -=1
                        iou_copy[max_r, :] = 0
                        iou_copy[:, max_c] = 0
                FN = n_targets - TP
                FP = n_predictions - TP
                self.TP.append(TP)
                self.FP.append(FP)
                self.FN.append(FN)

        else:
            for current_treshold in self.iou_tresholds:
                TP = 0
                FN = max(0, n_targets - n_predictions)
                FP = max(0, n_predictions - n_targets)
                self.TP.append(TP)
                self.FP.append(FP)
                self.FN.append(FN)


        F_beta = ((1 + self.beta**2) * np.sum(self.TP)) / ((1 + self.beta**2) * np.sum(self.TP) + np.sum(self.FP) + np.sum(self.FN) * self.beta**2)

        time_score = self.gamma * (max(0, self.tn - elapsed_time)/self.tn)

        self.mean_time_score.append(time_score)
        self.mean_fbetta_score = F_beta

        self.mean_score = (1 + np.mean(self.mean_time_score)) * F_beta

        return self.mean_score, np.mean(self.mean_time_score), F_beta


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

COLOR_VAL = [0., 1., 0.]

metric_2 = UpgreatMetric(iou_min=0.3, iou_max=1, iou_step=0.07, beta=1., gamma=0.15, tn=2., ignore=[0])

metric = MeanAveragePrecision(
      box_format='xywh',
      iou_thresholds=list(np.arange(0.3, 1, 0.07, dtype=np.float32)),
      iou_type='bbox',
)


def plot_results(pil_img, postprocessed_outputs, additions, width, height, target):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    # for output, addition in zip(postprocessed_outputs, additions):
    if postprocessed_outputs:
        scores = postprocessed_outputs['scores'].to('cpu')
        labels = postprocessed_outputs['labels'].to('cpu')
        boxes = postprocessed_outputs['boxes'].to('cpu')
    else:
        scores = labels = boxes = []

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


    pipeline_ = get_transformers_model(mlflow_uri, project_name, MODEL_NAME_VAL)
    model, image_processor = pipeline_.model, pipeline_.image_processor
    
    classifier = get_pytorch_model(mlflow_uri, project_name, CLASSIFIER_NAME, map_location=DEVICE, weights_only=False)

    image_processor.do_resize = False
    image_processor.do_normalize = False
    image_processor.do_pad = False

    dataset_path = VALIDATION_DATASET_PATH
    dataset = LizaDataset(dataset_path, image_processor=image_processor, transforms=None, training=False)

    model = model.to(DEVICE)
    if DO_COMPILE:
        import torch_tensorrt
        model = torch.compile(model, backend='torch_tensorrt',dynamic=False,
            options={
                "truncate_long_and_double": True,
                 # "precision": torch.half,
                 # "min_block_size": 2,
                 # "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
                 # "optimization_level": 5,
                 "use_python_runtime": False,
             }
         )

    testtime_transform = get_testtime_transforms()
    quantized = False
    pruned = False
    

    for n_image in tqdm(range(len(dataset))):
        t0 = time.time()
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
        # batch_size = INFERENCE_BATCH_SIZE
        len_subs = len(subs)
        batch_sizes = []
        for infer_size in INFERENCE_BATCH_SIZES:
            if len_subs > 0:
                div = len_subs // infer_size
                batch_sizes += [infer_size] * div
                len_subs -= div * infer_size
        
        # batch_sizes = [INFERENCE_BATCH_SIZE] * (len(subs) // INFERENCE_BATCH_SIZE) + len(subs)% INFERENCE_BATCH_SIZE * [1]
        batches = torch.split(subs, batch_sizes)
        batched_additions = torch.split(additions, batch_sizes)

        outputs_all = AttrDict()
        postprocessed_outputs_squeezed = AttrDict()

        for batch, addition in zip(batches, batched_additions):

                with torch.no_grad():
                    batch = batch.to(DEVICE)
                    outputs = model(batch)

                postprocessed_outputs_split = image_processor.post_process_object_detection(
                    outputs,
                    target_sizes=[(INFERENCE_SIZE, INFERENCE_SIZE)] * len(batch),
                    threshold=TRESHOLD,
                )

                for index, addi in enumerate(addition):
                    if len(postprocessed_outputs_split[index]['boxes']):
                        postprocessed_outputs_split[index]['boxes'][:,0] += addi[1]
                        postprocessed_outputs_split[index]['boxes'][:,1] += addi[0]
                        postprocessed_outputs_split[index]['boxes'][:,2] += addi[1]
                        postprocessed_outputs_split[index]['boxes'][:,3] += addi[0]

                postprocessed_outputs = AttrDict()
                for key in postprocessed_outputs_split[0].keys():
                    postprocessed_outputs[key] = torch.cat([out[key] for out in postprocessed_outputs_split])

                # postprocessed_outputs = postprocessed_outputs[0]

                if DO_TESTTIME_AUGMENT:
                    outputs_testtime = model(testtime_transform(batch))

                    postprocessed_outputs_testtime_split = image_processor.post_process_object_detection(
                        outputs_testtime,
                        target_sizes=[(INFERENCE_SIZE, INFERENCE_SIZE)] * len(batch),
                        threshold=TRESHOLD,
                    )


                    for index, addi in enumerate(addition):
                        if len(postprocessed_outputs_testtime_split[index]['boxes']):
                            
                            xmin = postprocessed_outputs_testtime_split[index]['boxes'][:, 0]
                            ymin = postprocessed_outputs_testtime_split[index]['boxes'][:, 1]
                            xmax = postprocessed_outputs_testtime_split[index]['boxes'][:, 2]
                            ymax = postprocessed_outputs_testtime_split[index]['boxes'][:, 3]
                    
                            postprocessed_outputs_testtime_split[index]['boxes'] = torch.stack(
                               [
                                   INFERENCE_SIZE - xmax,
                                   INFERENCE_SIZE - ymax,
                                   INFERENCE_SIZE - xmin,
                                   INFERENCE_SIZE - ymin,
                               ]
                            ).T

                            postprocessed_outputs_testtime_split[index]['boxes'][:,0] += addi[1]
                            postprocessed_outputs_testtime_split[index]['boxes'][:,1] += addi[0]
                            postprocessed_outputs_testtime_split[index]['boxes'][:,2] += addi[1]
                            postprocessed_outputs_testtime_split[index]['boxes'][:,3] += addi[0]

                    postprocessed_outputs_testtime = AttrDict()
                    for key in postprocessed_outputs_testtime_split[0].keys():
                        postprocessed_outputs_testtime[key] = torch.cat([out[key] for out in postprocessed_outputs_testtime_split])
                    # postprocessed_outputs_testtime = postprocessed_outputs_testtime[0]


                    iou = box_iou(postprocessed_outputs['boxes'], postprocessed_outputs_testtime['boxes'])

                    keep, _ = torch.where(iou > TESTTIME_IOU_TRESH)

                    # postprocessed_outputs = postprocessed_outputs_testtime
                    match TESTTIME_VARIANT:
                        case 'strict':
                            for key, value in postprocessed_outputs.items():
                                postprocessed_outputs[key] = value[keep]
                        case _:
                            for key in postprocessed_outputs.keys():
                                postprocessed_outputs[key] = torch.cat(
                                     [
                                         postprocessed_outputs[key],
                                         postprocessed_outputs_testtime[key],
                                     ],
                                     dim=0,
                                )

                # addition = addition[0]
                # if len(postprocessed_outputs['boxes']):
                #     postprocessed_outputs['boxes'][:,0] += addition[1]
                #     postprocessed_outputs['boxes'][:,1] += addition[0]
                #     postprocessed_outputs['boxes'][:,2] += addition[1]
                #     postprocessed_outputs['boxes'][:,3] += addition[0]
            
                for key, value in postprocessed_outputs.items():
                    if key not in postprocessed_outputs_squeezed:
                        postprocessed_outputs_squeezed[key] = []
                    [postprocessed_outputs_squeezed[key].append(val) for val in value]

        for key, value in postprocessed_outputs_squeezed.items():
            if len(value) > 0:
                postprocessed_outputs_squeezed[key] = torch.stack(value)
            else:
                postprocessed_outputs_squeezed[key] = torch.tensor([]).to(DEVICE)

        if DO_CLASSIFY:
            postprocessed_outputs_squeezed = reclassify(classifier, image, postprocessed_outputs_squeezed)
        if DO_NMS:
            postprocessed_outputs_squeezed = remove_overlaping(postprocessed_outputs_squeezed, NMS_IOU_TRESHOLD, ratio_tresh=RATIO_TRESH)
            
        outputs_for_comparison = AttrDict()
        if postprocessed_outputs_squeezed:
            for key, value in postprocessed_outputs_squeezed.items():
                if key == 'boxes' and len(value) > 0:
                    # value = value / torch.tensor([width, height, width, height]).to(DEVICE)
                    value = box_convert(value, 'xyxy', 'cxcywh')
                if key == 'score' and len(value) > 0:
                    value = value * 15
                outputs_for_comparison[key] = value
                # TODO: if saving convert to cxcywh

            targets_for_comparison = dict(
                 boxes=(inputs['labels']['boxes'] * torch.tensor([width, height, width, height])).to(DEVICE),
                 labels=inputs['labels']['class_labels'].to(DEVICE) + 1
            )

            metric.update(
                [outputs_for_comparison],
                [targets_for_comparison]
            )


        elapsed_time = time.time() - t0

        score, time_score, fbeta = metric_2.update(outputs_for_comparison, targets_for_comparison, elapsed_time)

        # os.system('clear')
        print(f'{n_image + 1} / {len(dataset)}')
        print('CURRENT MEAN SCORE:')
        print(f'Upgreat score: {score}')
        print(f'Upgreat fbeta: {fbeta}')
        print(f'Upgreat time score: {1 + time_score}')
        pprint(metric.compute())

        if VERBOSE:
            plot_results(image.numpy().transpose((1, 2, 0)), postprocessed_outputs_squeezed, additions, width, height, inputs['labels']['boxes'])

    os.system('clear')
    print('FINAL SCORE:')
    pprint(metric.compute())
