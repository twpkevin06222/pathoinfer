"""Nucleus instances from HoVer-Net style outputs: the watershed and each instance's type.

Ported from pathotrain's ``cellseg/instances.py`` (Apache-2.0, same authors), which
validates its models with this code, so inference reproduces training-time results.
Algorithm: HoVer-Net, Graham et al. 2019, "HoVer-Net: Simultaneous segmentation and
classification of nuclei in multi-tissue histology images" (written from the paper).

Inputs are one tile's maps as NumPy arrays: the nucleus probability ``(H, W)``, the HV
maps ``(2, H, W)`` and the type logits ``(C, H, W)``. An instance map is ``(H, W)``
int32, 0 for background and one id per nucleus.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.segmentation import watershed

# HoVer-Net takes HV-map gradients with a 21 px Sobel kernel (cv2), whose smoothing is a
# binomial of variance 20 / 4 = 5: here a Gaussian derivative of that sigma. The maps
# are min-max scaled afterwards, so the kernel's scale does not matter.
SOBEL_SIGMA = math.sqrt(5)
DISK = np.hypot(*np.mgrid[-2:3, -2:3]) <= 2.5  # 5 x 5 ellipse, HoVer-Net's marker opening


def softmax(logits: NDArray[np.floating], axis: int = 0) -> NDArray[np.float32]:
    shifted = np.exp(logits - logits.max(axis, keepdims=True))
    return (shifted / shifted.sum(axis, keepdims=True)).astype(np.float32)


def _minmax(x: NDArray[np.floating]) -> NDArray[np.floating]:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def _remove_small(labels: NDArray[np.integer], min_size: int) -> NDArray[np.integer]:
    sizes = np.bincount(labels.ravel())
    small = sizes < min_size
    small[0] = False
    return np.where(small[labels], 0, labels)


def watershed_instances(
    foreground: NDArray[np.floating],
    hv: NDArray[np.floating],
    threshold: float = 0.5,
    marker_threshold: float = 0.4,
    min_size: int = 10,
) -> NDArray[np.int32]:
    """HoVer-Net's watershed: an instance map from the nucleus probability ``(H, W)`` and
    the HV maps ``(2, H, W)``.

    Nuclei are where ``foreground >= threshold``. The HV maps rise across a nucleus and fall
    steeply between two touching ones, so their most negative gradients mark boundaries;
    nucleus pixels away from them (gradient energy below ``marker_threshold``) seed a
    watershed over the nuclei. Blobs and seeds under ``min_size`` pixels are dropped.
    """
    blobs, _ = ndi.label(foreground >= threshold)
    nuclei = _remove_small(blobs, min_size) > 0
    if not nuclei.any():
        return np.zeros(foreground.shape, np.int32)
    h, v = _minmax(hv[0]), _minmax(hv[1])
    grad_h = 1 - _minmax(ndi.gaussian_filter(h, SOBEL_SIGMA, order=(0, 1)))
    grad_v = 1 - _minmax(ndi.gaussian_filter(v, SOBEL_SIGMA, order=(1, 0)))
    energy = np.clip(np.maximum(grad_h, grad_v) - ~nuclei, 0, None)
    # HoVer-Net blurs with a 3 px kernel (cv2 sigma 0.8 for that size).
    dist = -ndi.gaussian_filter((1 - energy) * nuclei, 0.8)
    markers = ndi.binary_fill_holes(nuclei & (energy < marker_threshold))
    markers, _ = ndi.label(ndi.binary_opening(markers, DISK))
    markers = _remove_small(markers, min_size).astype(np.int32)
    flooded = watershed(dist, markers, mask=nuclei)  # type: ignore[no-untyped-call]  # skimage
    return np.asarray(flooded, np.int32)


def instance_types(
    instances: NDArray[np.integer], type_logits: NDArray[np.floating]
) -> NDArray[np.int64]:
    """The type of each instance id, indexed by id (``out[0]`` is 0): the most common
    per-pixel argmax of ``type_logits`` ``(C, H, W)`` under it, among types 1..C-1;
    0 when no pixel of it has one."""
    n_types = type_logits.shape[0]
    size = int(instances.max()) + 1
    labels = type_logits.argmax(0)
    counts = np.bincount(
        instances.ravel().astype(np.int64) * n_types + labels.ravel(),
        minlength=size * n_types,
    ).reshape(size, n_types)
    counts[:, 0] = 0  # background votes only when nothing else does
    out = counts.argmax(1).astype(np.int64)
    out[0] = 0
    return out
