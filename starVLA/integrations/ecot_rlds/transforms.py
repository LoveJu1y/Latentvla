# SPDX-License-Identifier: MIT
"""
Image and modality transforms for the ECOT RLDS adapter.

Handles conversion from RLDS numpy arrays to PIL images and resizing.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image


def to_pil_and_resize(img_array: np.ndarray, target_size: Tuple[int, int]) -> Image.Image:
    """
    Convert numpy array to PIL.Image and resize.

    Parameters
    ----------
    img_array:
        [H, W, 3] uint8 array (or float array in [0, 1])
    target_size:
        (H, W) target size tuple

    Returns
    -------
    PIL.Image.Image
        Resized PIL image with size (W, H) (PIL convention)
    """
    # Ensure uint8 dtype
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            # Assume normalized float [0, 1]
            img_array = (img_array * 255).astype(np.uint8)
        else:
            # Assume float in [0, 255]
            img_array = img_array.astype(np.uint8)

    # Convert to PIL
    pil_img = Image.fromarray(img_array)

    # PIL.Image.resize expects (width, height)
    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    return pil_img


def to_pil_list_from_numpy(primary: np.ndarray) -> List[Image.Image]:
    """
    Convert the primary camera frames from numpy arrays to PIL images.

    Handles RLDS format: [window_size, H, W, 3] or [H, W, 3].
    Only extracts the first window (current frame) when window_size > 1.

    Note: This function only does minimal conversion (numpy -> PIL).
    Image resize/normalize will be handled by Qwen processor during training.
    If RLDS frame_transform has resize_size set, images should already be resized.

    Parameters
    ----------
    primary:
        Numpy array representing the primary camera observations.
        Expected shapes:
        - [window_size, H, W, 3] (RLDS format with window)
        - [H, W, 3] (single frame)

    Returns
    -------
    list[PIL.Image.Image]
        A list containing PIL images (currently single primary view).
        Can be extended to include multiple views (wrist, etc.) in the future.
    """
    if primary.size == 0:
        raise ValueError("Received empty primary image array.")

    # Handle different input shapes
    if primary.ndim == 4:
        # [window_size, H, W, 3] - take first window (current frame)
        frame = primary[0]  # [H, W, 3]
    elif primary.ndim == 3:
        # [H, W, 3] - single frame
        frame = primary
    else:
        raise ValueError(
            f"Unexpected image array shape: {primary.shape}. "
            f"Expected [window, H, W, 3] or [H, W, 3]"
        )

    # Convert to PIL - minimal processing, let Qwen processor handle resize/normalize
    # Ensure uint8 dtype for PIL conversion
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            # Normalized float [0, 1] -> uint8 [0, 255]
            frame = (frame * 255).astype(np.uint8)
        else:
            # Float in [0, 255] -> uint8
            frame = np.clip(frame, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(frame)

    # Return as list (for multi-view extension)
    return [pil_img]


def ensure_image_size(images: Iterable[Image.Image], image_size: Tuple[int, int]) -> List[Image.Image]:
    """
    Resize images to the configured size if necessary.

    Note: This is mainly for consistency with LeRobot dataset behavior.
    If RLDS frame_transform has resize_size set correctly, images should already
    be the right size. Qwen processor will handle final normalization and any
    additional processing needed for the model.

    Parameters
    ----------
    images:
        Iterable of PIL images.
    image_size:
        Target (height, width) tuple.

    Returns
    -------
    list[PIL.Image.Image]
        Resized images. Each image will have size (W, H) matching image_size.
    """
    resized = []
    target_size_pil = (image_size[1], image_size[0])  # PIL uses (width, height)

    for img in images:
        if img.size != target_size_pil:
            resized.append(img.resize(target_size_pil, Image.BILINEAR))
        else:
            resized.append(img)

    return resized

