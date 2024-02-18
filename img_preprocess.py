from typing import Tuple

import cv2
import numpy as np
import tqdm
from cv2.typing import Size, Rect
from tqdm import tqdm


def resize_and_pad_naive(img: np.ndarray, dst_size: Size) -> np.ndarray:
    origin_height, origin_width = img.shape[:2]
    dst_width, dst_height = dst_size

    # keep aspect ratio
    scale = min(dst_width / origin_width, dst_height / origin_height)
    new_width = int(origin_width * scale)
    new_height = int(origin_height * scale)
    resized_img = cv2.resize(img, (new_width, new_height))

    x_offset = (dst_width - new_width) // 2
    y_offset = (dst_height - new_height) // 2

    out = np.zeros([dst_height, dst_width, 3], dtype=np.uint8)
    out[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
    return out


def crop_resize_and_pad_naive(img: np.ndarray, box: Rect, dst_size: Size) -> np.ndarray:
    img = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    return resize_and_pad_naive(img, dst_size)


def resize_and_pad(img: np.ndarray, dst_size: Size) -> Tuple[np.ndarray, np.ndarray]:
    sw, sh = img.shape[1], img.shape[0]
    dw, dh = dst_size
    scale = min(dw / sw, dh / sh)
    s_inv = 1 / scale

    scx = sw * 0.5
    scy = sh * 0.5

    dcx = dst_size[0] * 0.5
    dcy = dst_size[1] * 0.5

    x_offset = dcx - scx * scale
    y_offset = dcy - scy * scale

    mat = np.array([
        [scale, 0, x_offset],
        [0, scale, y_offset],
    ])

    inv_mat = np.array([
        [s_inv, 0, - x_offset * s_inv],
        [0, s_inv, - y_offset * s_inv],
    ])

    out = cv2.warpAffine(img, mat, dst_size)
    return out, inv_mat


def crop_resize_and_pad(img: np.ndarray, box: Rect, dst_size: Size) -> Tuple[np.ndarray, np.ndarray]:
    sw, sh = box[2:]
    dw, dh = dst_size
    scale = min(dw / sw, dh / sh)
    s_inv = 1 / scale

    scx = box[0] + box[2] * 0.5
    scy = box[1] + box[3] * 0.5

    dcx = dst_size[0] * 0.5
    dcy = dst_size[1] * 0.5

    x_offset = dcx - scx * scale
    y_offset = dcy - scy * scale

    mat = np.array([
        [scale, 0, x_offset],
        [0, scale, y_offset],
    ])

    inv_mat = np.array([
        [s_inv, 0, - x_offset * s_inv],
        [0, s_inv, - y_offset * s_inv],
    ])

    out = cv2.warpAffine(img, mat, dst_size)
    return out, inv_mat


def crop_resize_and_pad_blank_border(img: np.ndarray, box: Rect, dst_size: Size) -> Tuple[np.ndarray, np.ndarray]:
    img = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    out, inv_mat = resize_and_pad(img, dst_size)

    # add x,y offset to inv_mat
    inv_mat[0, 2] += box[0]
    inv_mat[1, 2] += box[1]

    return out, inv_mat


def test_function():
    img = cv2.imread("test.jpg")

    dst_size = (384, 256)
    roi_box = (100, 200, 200, 100)

    processed, _, = crop_resize_and_pad_blank_border(img, roi_box, dst_size)
    cv2.imwrite("test_processed.jpg", processed)

    processed_naive = crop_resize_and_pad_naive(img, roi_box, dst_size)
    cv2.imwrite("test_processed_naive.jpg", processed_naive)

    cv2.rectangle(img, roi_box[:2], (roi_box[0] + roi_box[2], roi_box[1] + roi_box[3]), (225, 0, 0), 10)
    pass


def test_speed():
    test_img = np.random.randint(0, 255, size=[1080, 1920, 3], dtype=np.uint8)
    N = 100000

    for _ in tqdm(range(N), desc="Affine"):
        crop_resize_and_pad_blank_border(test_img, (100, 200, 300, 400), (256, 192))

    for _ in tqdm(range(N), desc="Naive"):
        crop_resize_and_pad_naive(test_img, (100, 200, 300, 400), (256, 192))


test_function()
test_speed()
