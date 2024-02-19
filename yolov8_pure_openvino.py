from tqdm import tqdm
from typing import Tuple

import torch
import numpy as np
import cv2
from pathlib import Path

from openvino import Core, Type, Layout, CompiledModel
from openvino.preprocess import PrePostProcessor, ColorFormat
from torchvision.ops import batched_nms
from ultralytics import YOLO

from img_preprocess import resize_and_pad, inv_points


def export_openvino():
    if not Path("yolov8n_openvino_model").exists():
        model = YOLO("yolov8n.pt")
        model.export(format="openvino")


def compile_model():
    ov_core = Core()
    ov_model = ov_core.read_model("yolov8n_openvino_model/yolov8n.xml")

    ppp = PrePostProcessor(ov_model)
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout("NHWC")) \
        .set_color_format(ColorFormat.BGR)

    ppp.input().preprocess() \
        .convert_element_type(Type.f32) \
        .convert_layout(Layout("NCHW")) \
        .scale(255)

    print(ppp)
    ov_model = ppp.build()
    return ov_core.compile_model(ov_model)


def visualize_det(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
    img = img.copy()
    for bbox, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = bbox.astype(np.int64)
        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Calculate text position
        text_x = x_min + 10
        text_y = y_min - 10

        # Draw label
        cv2.putText(img, str(label), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img


def detect_image(model: CompiledModel, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img, inv_mat = resize_and_pad(image, (640, 640), (114, 114, 114))
    ov_out = model(np.expand_dims(img, 0))[0]

    det = ov_out[0].T

    # cxcywh to xyxy
    det[:, 0] -= det[:, 2] * 0.5
    det[:, 1] -= det[:, 3] * 0.5
    det[:, 2] += det[:, 0]
    det[:, 3] += det[:, 1]

    boxes = det[:, :4]
    scores = det[:, 4:]
    cls = np.argmax(scores, axis=1)
    scores = scores[np.arange(8400), cls]
    score_masks = scores > 0.25

    boxes = boxes[score_masks]
    scores = scores[score_masks]
    cls = cls[score_masks]

    kept_idx = batched_nms(
        torch.as_tensor(boxes),
        torch.as_tensor(scores),
        torch.as_tensor(cls),
        0.7
    )

    boxes = boxes[kept_idx]
    scores = scores[kept_idx]
    cls = cls[kept_idx]

    # inv points x,y
    boxes = inv_points(boxes.reshape(-1, 2), inv_mat).reshape(-1, 4)

    return boxes, cls, scores


def test_function():
    export_openvino()
    model = compile_model()
    org_img = cv2.imread("test.jpg")
    boxes, cls, _ = detect_image(model, org_img)
    det_res = visualize_det(org_img, boxes, cls)


def test_speed():
    N = 1000
    export_openvino()
    org_img = cv2.imread("test.jpg")

    model = compile_model()
    for _ in tqdm(range(N), desc="Pure"):
        detect_image(model, org_img)

    model = YOLO("yolov8n_openvino_model")
    for _ in tqdm(range(N), desc="Naive"):
        model(org_img, verbose=False)


def main():
    test_function()
    test_speed()


if __name__ == '__main__':
    main()
