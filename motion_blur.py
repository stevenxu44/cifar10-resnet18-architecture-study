"""Deterministic motion-blur kernels for training and evaluation."""
import cv2
import numpy as np
from PIL import Image, ImageFilter


def motion_kernel(length, direction):
    if length not in (3, 5, 7):
        raise ValueError("Supported lengths: 3, 5, 7")
    if direction not in ("horizontal", "vertical", "diag_down", "diag_up",
                         "shallow_down"):
        raise ValueError(f"Unknown direction: {direction}")
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    if direction == "horizontal":
        kernel[center, :] = 1
    elif direction == "vertical":
        kernel[:, center] = 1
    elif direction == "diag_down":
        np.fill_diagonal(kernel, 1)
    elif direction == "diag_up":
        np.fill_diagonal(np.fliplr(kernel), 1)
    else:
        for x in range(length):
            y = center + round(0.5 * (x - center))
            kernel[y, x] = 1
    return kernel / kernel.sum()


def apply_motion(image, length, direction):
    kernel = motion_kernel(length, direction)
    if length in (3, 5):
        return image.filter(ImageFilter.Kernel((length, length),
                                             kernel.ravel().tolist(), scale=1))
    array = cv2.filter2D(np.asarray(image), -1, kernel,
                         borderType=cv2.BORDER_REFLECT_101)
    return Image.fromarray(array)


def motion_images(images, length, direction):
    return np.stack([np.asarray(apply_motion(Image.fromarray(image), length, direction))
                     for image in images])
