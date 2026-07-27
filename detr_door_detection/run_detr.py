import matplotlib.pyplot as plt
import torch
from torch import nn
from typing import Type
import cv2
import numpy as np
from time import time
import os
from pathlib import Path
from detr_door_detection.utils import *
import torch.nn.functional as F

DESCRIPTION = int
title = 'QD75'
COLORS = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
DEEP_DOORS_2_LABELLED_EXP: DESCRIPTION = 0
EXP_1_HOUSE_1: DESCRIPTION = 1
EXP_2_HOUSE_1_25: DESCRIPTION = 2
EXP_2_HOUSE_1_50: DESCRIPTION = 3
EXP_2_HOUSE_1_75: DESCRIPTION = 4
EXP_1_HOUSE_2: DESCRIPTION = 5
EXP_2_HOUSE_2_25: DESCRIPTION = 6
EXP_2_HOUSE_2_50: DESCRIPTION = 7
EXP_2_HOUSE_2_75: DESCRIPTION = 8
EXP_1_HOUSE_7: DESCRIPTION = 9
EXP_2_HOUSE_7_25: DESCRIPTION = 10
EXP_2_HOUSE_7_50: DESCRIPTION = 11
EXP_2_HOUSE_7_75: DESCRIPTION = 12
EXP_1_HOUSE_9: DESCRIPTION = 13
EXP_2_HOUSE_9_25: DESCRIPTION = 14
EXP_2_HOUSE_9_50: DESCRIPTION = 15
EXP_2_HOUSE_9_75: DESCRIPTION = 16
EXP_1_HOUSE_10: DESCRIPTION = 17
EXP_2_HOUSE_10_25: DESCRIPTION = 18
EXP_2_HOUSE_10_50: DESCRIPTION = 19
EXP_2_HOUSE_10_75: DESCRIPTION = 20
EXP_1_HOUSE_13: DESCRIPTION = 21
EXP_2_HOUSE_13_25: DESCRIPTION = 22
EXP_2_HOUSE_13_50: DESCRIPTION = 23
EXP_2_HOUSE_13_75: DESCRIPTION = 24
EXP_1_HOUSE_15: DESCRIPTION = 25
EXP_2_HOUSE_15_25: DESCRIPTION = 26
EXP_2_HOUSE_15_50: DESCRIPTION = 27
EXP_2_HOUSE_15_75: DESCRIPTION = 28
EXP_1_HOUSE_20: DESCRIPTION = 29
EXP_2_HOUSE_20_25: DESCRIPTION = 20
EXP_2_HOUSE_20_50: DESCRIPTION = 31
EXP_2_HOUSE_20_75: DESCRIPTION = 32
EXP_1_HOUSE_21: DESCRIPTION = 33
EXP_2_HOUSE_21_25: DESCRIPTION = 34
EXP_2_HOUSE_21_50: DESCRIPTION = 35
EXP_2_HOUSE_21_75: DESCRIPTION = 36
EXP_1_HOUSE_22: DESCRIPTION = 37
EXP_2_HOUSE_22_25: DESCRIPTION = 38
EXP_2_HOUSE_22_50: DESCRIPTION = 39
EXP_2_HOUSE_22_75: DESCRIPTION = 40

FINAL_DOORS_DATASET = 'final_doors_dataset'
DETR_RESNET50 = 'detr_resnet50'
ModelName = str
DETR_RESNET50: ModelName = 'detr_resnet50'
DATASET = Type[str]



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class DetrDoorDetector(nn.Module):
    """
    This class builds a door detector starting from a detr pretrained module.
    Basically it loads a dtr module and modify its structure to recognize door.
    """
    def __init__(
        self,
        model_name: ModelName,
        n_labels: int,
        pretrained: bool,
        dataset_name: DATASET,
        description: DESCRIPTION,
        detr_source_dir=None,
    ):
        """

        :param model_name: the name of the detr base model
        :param n_labels: the labels' number
        :param pretrained: it refers to the DetrDoorDetector class, not to detr base model.
                            It True, the DetrDoorDetector's weights are loaded, otherwise the weights are loaded only for the detr base model
        """
        super(DetrDoorDetector, self).__init__()
        self._model_name = model_name
        repo_root = Path(__file__).resolve().parents[1]
        source_dir = Path(
            detr_source_dir
            or os.environ.get("ACTIVE_ROOM_DETR_DIR", repo_root / "third_party" / "detr")
        ).expanduser().resolve()
        if not (source_dir / "hubconf.py").is_file():
            raise FileNotFoundError(
                "Pinned DETR source is missing: {}. Run scripts/bootstrap_modern.sh first.".format(
                    source_dir
                )
            )
        self.model = torch.hub.load(
            str(source_dir),
            model_name,
            source="local",
            pretrained=False,
        )
        self._dataset_name = dataset_name
        self._description = description

        # Change the last part of the model
        self.model.query_embed = nn.Embedding(10, self.model.transformer.d_model)
        self.model.class_embed = nn.Linear(256, n_labels + 1)

        if pretrained:
            path = os.path.join(os.path.dirname(__file__), 'train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            weights_path = os.path.join(path, 'model.pth')
            if not os.path.isfile(weights_path):
                raise FileNotFoundError("Door detector checkpoint is missing: {}".format(weights_path))
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model(x)

        """
        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape=[batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the un-normalized bounding box.
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        return x

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        super().train(mode)
        return self

    def to(self, device):
        return super().to(device)

    def save(self, epoch, optimizer_state_dict, lr_scheduler_state_dict, params, logs):
        path = os.path.join(os.path.dirname(__file__), 'train_params', self._model_name + '_' + str(self._description))

        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, str(self._dataset_name))

        if not os.path.exists(path):
            os.mkdir(path)

        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
        torch.save(
            {
                'optimizer_state_dict': optimizer_state_dict,
                'lr_scheduler_state_dict': lr_scheduler_state_dict
            }, os.path.join(path, 'checkpoint.pth')
        )

        torch.save(
            {
                'epoch': epoch,
                'logs': logs,
                'params': params,
            }, os.path.join(path, 'training_data.pth')
        )

    def load_checkpoint(self,):
        path = os.path.join(os.path.dirname(__file__), 'train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
        checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))
        training_data = torch.load(os.path.join(path, 'training_data.pth'))

        return {**checkpoint, **training_data}

    def set_description(self, description: DESCRIPTION):
        self._description = description


_model = None
_model_device = None


def _resolve_device(device=None):
    requested = device or os.environ.get("ACTIVE_ROOM_DETECTOR_DEVICE", "cuda")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Door detector device must be either 'cpu' or 'cuda'")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA door detection was requested, but CUDA is unavailable")
    return resolved


def get_door_detector(device=None):
    global _model
    global _model_device

    resolved_device = _resolve_device(device)
    if _model is None:
        _model = DetrDoorDetector(
            model_name=DETR_RESNET50,
            n_labels=2,
            pretrained=True,
            dataset_name=FINAL_DOORS_DATASET,
            description=EXP_2_HOUSE_1_75,
        )
    if _model_device != resolved_device:
        _model.to(resolved_device)
        _model_device = resolved_device
    _model.eval()
    return _model, resolved_device


def run_detr(img, device=None):  # 256, 256, 3
    global title
    model, model_device = get_door_detector(device)
    model.eval()
    img_vis = img.copy()
    image_height, image_width = img.shape[:2]
    img_cv2 = np.zeros((image_height, image_width), np.uint8)
    img_mask_full = np.zeros((image_height, image_width), np.uint8)
    input_tensor = torch.from_numpy(
        np.ascontiguousarray(img.transpose(2, 0, 1))
    ).unsqueeze(0).to(device=model_device, dtype=torch.float32)
    with torch.inference_mode():
        outputs = [(model(input_tensor), title)]
    # t2 = time()
    post_processor = PostProcess().to(model_device)
    img_size = [image_height, image_width]
    target_sizes = torch.tensor([img_size], device=model_device)
    processed_data_models = [(post_processor(outputs=output, target_sizes=target_sizes), title) for
                             output, title in outputs]

    post_processed_data = processed_data_models[0]
    image_data = {
        key: value.detach().cpu()
        for key, value in post_processed_data[0][0].items()
    }
    title = post_processed_data[1]
    keep = image_data['scores'] > 0.5
    for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep],
                                                      image_data['boxes'][keep]):

        label = label.item()

        if label == 0:
            break
        if score.item() < 0.85:  # 0.9
            break

        # label 0 close, label 1 open
        (xmin, ymin, xmax, ymax) = np.rint(
            np.array([xmin, ymin, xmax, ymax])
        ).astype('int32')

        img_cv2 = cv2.rectangle(img_cv2, (xmin, (ymin+ymax)//2-20), (xmax, (ymin+ymax)//2), #(xmin, (ymin+ymax)//2-20), (xmax, (ymin+ymax)//2),
                                color=1, thickness=(xmax-xmin)//4)  # 20 (xmax-xmin)//4
        """img_cv2 = cv2.line(img_cv2, (xmin, (ymin + ymax) // 2 - 20), (xmin, (ymin + ymax) // 2),
                                color=1, thickness=(xmax - xmin) // 4)  # 20 (xmax-xmin)//4
        img_cv2 = cv2.line(img_cv2, (xmax, (ymin + ymax) // 2 - 20), (xmax, (ymin + ymax) // 2),
                           color=1, thickness=(xmax - xmin) // 4)  # 20 (xmax-xmin)//4"""
        #img_mask_full = cv2.rectangle(img_mask_full, (max(0,xmin-(xmax-xmin)//3), ymin), (min(img.shape[1],xmax+(xmax-xmin)//3), ymax),
        #                        color=1, thickness=-1)  # (max(0,xmin-(xmax-xmin)//3), ymin), (min(img.shape[1],xmax+(xmax-xmin)//3), ymax)

        #img_vis = cv2.rectangle(img_vis, (xmin, (ymin+ymax)//2-20), (xmax, (ymin+ymax)//2),
        #                        color=(np.array(COLORS[label]) * 255)[::-1].astype('int16').tolist(), thickness=(xmax-xmin)//4)
        """img_vis = cv2.line(img_vis, (xmin, (ymin + ymax) // 2 - 20), (xmin, (ymin + ymax) // 2),
                                    color=(np.array(COLORS[label]) * 255)[::-1].astype('int16').tolist(), thickness=(xmax-xmin)//4)
        img_vis = cv2.line(img_vis, (xmax, (ymin + ymax) // 2 - 20), (xmax, (ymin + ymax) // 2),
                           color=(np.array(COLORS[label]) * 255)[::-1].astype('int16').tolist(),
                           thickness=(xmax - xmin) // 4)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        #img_vis = cv2.putText(img_vis, '{:.3f}'.format(score), (xmin + 10, ymin + 10), font, 0.5, (0, 0, 255), 1)
    img_mask_full = cv2.rectangle(img_mask_full, (0, 0),
                                  (image_width, image_height*3//5),
                                  color=1, thickness=-1)
    return img_cv2, img_vis, img_mask_full

if __name__ == '__main__':
    pic_list = os.listdir('/home/airs/Downloads/door_detection/door-detection-long-term/doors_detection_long_term/doors_detector/pic_sampled/')

    img_path = '/home/airs/Downloads/door_detection/door-detection-long-term/doors_detection_long_term/doors_detector/pic_sampled/1.png'
    img = plt.imread(img_path)
    print(img)
    result = run_detr(img)
    plt.imshow(result)
    plt.show()
