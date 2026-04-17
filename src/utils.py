"""
utils.py
--------
Shared image-processing and ROI preparation utilities.
"""

import numpy as np
import cv2
import os
import pandas as pd
import torch
import torch.nn.functional as F
import time
from PIL import Image


__all__ = [
    'get_boundary_pid',
    'max_num',
    'preprocess',
    'crop_circle_with_mask_float',
    'rotate_image',
    'crop_top_center_square',
    'append_ij_angle_to_pdata',
    'get_disk_img',
    'iter_contact_roi_batches',
    'gaussian_kernel',
    'smooth_image',
    'generate_cropped_batch',
    'dog_kernel',
    'crop_tangent_square',
    'to_pil_uint8',
    'read_image_with_retry'
]

def to_pil_uint8(img):
    img = np.asarray(img)
    if img.max() <= 1.0:
        img8 = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img8 = img.clip(0, 255).astype(np.uint8)
    return Image.fromarray(img8)

def get_boundary_pid(f, xmin, xmax, ymin, ymax):
    mask = (
        (f['x'] < xmin) | (f['x'] > xmax) |
        (f['y'] < ymin) | (f['y'] > ymax)
    )
    boundary_pid = f.particle[mask].to_numpy()
    return boundary_pid

def max_num(img_dir: str, exp_folder: str, prefix: str) -> int:
    """Return the highest image index found in the experiment folder.

    Parameters
    ----------
    img_dir    : root image directory
    exp_folder : experiment sub-folder name
    prefix     : filename prefix, e.g. 'bw_', 'green_', 'blue_'
    """
    files = os.listdir(os.path.join(img_dir, exp_folder))
    frame_numbers = [
        int(f.split('_')[1].split('.')[0])
        for f in files
        if f.startswith(prefix) and f.endswith('.png')
    ]
    return int(np.max(frame_numbers)) if frame_numbers else 0


def crop_circle_with_mask_float(img: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the inscribed circle of a square float image.

    Parameters
    ----------
    img : (H, W, C) float array – assumed H == W

    Returns
    -------
    Masked image with the same shape and dtype as *img*.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])

    if img.ndim == 2:
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= radius ** 2
    else:
        y_grid, x_grid, _ = np.ogrid[:h, :w, : img.shape[2]]
        mask = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= radius ** 2

    return img * mask


def preprocess(image, radius):
    """Convert a BGR crop to blurred, normalised grayscale at training size."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target = (80, 80) if radius == 37 else (100, 100)
    gray = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(gray, (3, 3), 1).astype(np.float32)
    max_val = blurred.max()
    return blurred / max_val if max_val > 0 else blurred


def rotate_image(image, angle_degrees):
    """Rotate an image around its center using zero padding outside the frame."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle_degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_top_center_square(image, center, radius):
    """Crop a top-centered square region from a rotated contact image."""
    h, w = image.shape[:2]
    side = int(radius)
    x1 = max(center[0] - side // 2, 0)
    x2 = min(center[0] + side // 2, w)
    return image[0:min(side, h), x1:x2]


def append_ij_angle_to_pdata(pdata):
    """Add a beta column containing the angle of the i→j vector."""
    pdata = pdata.copy()
    dx = pdata['xj'].to_numpy() - pdata['xi'].to_numpy()
    dy = pdata['yj'].to_numpy() - pdata['yi'].to_numpy()
    pdata['beta'] = np.arctan2(dy, dx)
    return pdata


def get_disk_img(pdata, img):
    """Crop, preprocess, and circular-mask the particle-i disk image."""
    img = img[250:1200, :]
    entry = pdata.copy()
    if isinstance(entry, pd.DataFrame) and len(entry) > 1:
        entry = entry.iloc[0]

    x = int(np.around(entry['xi']))
    y = int(np.around(entry['yi']))
    r = int(entry['ri'])
    h, w = img.shape[:2]
    y1, y2 = max(0, y - r), min(h, y + r)
    x1, x2 = max(0, x - r), min(w, x + r)
    img_crop = img[y1:y2, x1:x2]

    if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
        raise ValueError('Cropped region is empty - check particle position or radius.')

    return crop_circle_with_mask_float(preprocess(img_crop, r))


def iter_contact_roi_batches(F_bond, IMG_DIR, frame_lag=0, batch_size=256, progress_every=1):
    """Yield contact ROI images/metadata in batches to avoid large peak memory."""
    from IPython.display import clear_output

    F_bond = append_ij_angle_to_pdata(F_bond)
    grouped = F_bond.groupby('frame')

    batch_images = []
    batch_metadata = []
    total_seen = 0

    max_frame = int(F_bond.frame.max()) if len(F_bond) else 0
    for frame in range(1, max_frame + 1):
        if frame % progress_every == 0:
            clear_output(wait=True)
            print(f'Cropping contact ROIs — frame: {frame}/{max_frame}')

        if frame not in grouped.groups:
            continue

        img_path = os.path.join(IMG_DIR, f'bw_{frame + frame_lag}.png')
        img = cv2.imread(img_path)
        if img is None:
            print(f'Warning: could not read image {img_path}; skipping frame {frame}')
            continue

        frame_data = grouped.get_group(frame)
        for idx, pdata in frame_data.iterrows():
            try:
                img_crop = get_disk_img(pdata, img)
            except ValueError as err:
                print(f'Skipping contact {idx}: {err}')
                continue

            img_masked = crop_circle_with_mask_float(img_crop)
            radius = int(pdata['ri'])
            center = (img_crop.shape[1] // 2, img_crop.shape[0] // 2)
            img_rotated = rotate_image(img_masked, pdata['beta'] * 180 / np.pi + 90)
            img_final = crop_top_center_square(img_rotated, center, radius)

            batch_images.append(img_final)
            batch_metadata.append({
                'idx': idx,
                'frame': frame,
                'beta': pdata['beta'],
                'xi': pdata['xi'],
                'yi': pdata['yi'],
                'ri': pdata['ri'],
            })
            total_seen += 1

            if len(batch_images) >= batch_size:
                yield batch_images, batch_metadata, total_seen
                batch_images, batch_metadata = [], []

        del img

    if batch_images:
        yield batch_images, batch_metadata, total_seen

    clear_output(wait=True)


def gaussian_kernel(kernel_size=3, sigma=1.0, device='cuda'):
    """Create a normalized 2-D Gaussian kernel on the target torch device."""
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    g /= g.sum()
    return g[:, None] @ g[None, :]


def smooth_image(img, kernel_size=3, sigma=1.0):
    """Apply a 2-D Gaussian blur to a single-channel [H, W] torch tensor."""
    device = img.device
    kernel = gaussian_kernel(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0)
    return F.conv2d(
        img.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=kernel_size // 2,
    ).squeeze(0).squeeze(0)


def generate_cropped_batch(f: "pd.DataFrame", img: np.ndarray) -> np.ndarray:
    """Crop, resize, blur, and normalise one image patch per particle row in *f*.

    Parameters
    ----------
    f   : DataFrame with columns x, y, rpx
    img : full-frame BGR image

    Returns
    -------
    batch : float32 array of shape (len(f), 128, 128, 3), values in [0, 1]
    """
    batch = np.empty((len(f), 128, 128, 3), dtype=np.float32)
    h, w = img.shape[:2]
    for i in range(len(f)):
        row = f.iloc[i]
        x, y, r = int(np.around(row['x'])), int(np.around(row['y'])), int(row['rpx'])
        y1, y2 = max(0, y - r), min(h, y + r)
        x1, x2 = max(0, x - r), min(w, x + r)
        patch = img[y1:y2, x1:x2]
        patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA)
        patch = cv2.blur(patch, ksize=(3, 3)).astype(np.float32) / 255
        patch = crop_circle_with_mask_float(patch)
        batch[i] = patch
    return batch


def dog_kernel(r_center: int, delta: int, sigma: int) -> np.ndarray:
    """Create a radial Difference-of-Gaussians kernel for circular edge detection.

    Parameters
    ----------
    r_center : expected radius of the disk (pixels)
    delta    : half-width of the annular band
    sigma    : Gaussian std-dev for the two rings

    Returns
    -------
    kernel : float32 array, normalised to ±1
    """
    yy_k, xx_k = np.ogrid[
        -r_center - 2 * sigma : r_center + 2 * sigma + 1,
        -r_center - 2 * sigma : r_center + 2 * sigma + 1,
    ]
    rr_k = np.sqrt(xx_k ** 2 + yy_k ** 2)

    inner = -np.exp(-((rr_k - (r_center - delta)) ** 2) / (2 * sigma ** 2))
    outer =  np.exp(-((rr_k - (r_center + delta)) ** 2) / (2 * sigma ** 2))

    kernel = inner + outer
    kernel /= np.max(np.abs(kernel))
    return kernel.astype(np.float32)

def crop_tangent_square(img, center, angle_rad, crop_size):
    """Crop a rotated ROI so the contact lies at the bottom of the image."""
    M = cv2.getRotationMatrix2D(center, np.degrees(angle_rad) - 90, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    x, y = int(center[0]), int(center[1])
    half = crop_size // 2
    return rotated[y - half:y + half, x - half:x + half]


def read_image_with_retry(image_path, retries=4, delay_s=0.15):
    """Read image robustly. Handles transient cv2.imread(None) returns on busy I/O."""
    if not os.path.exists(image_path):
        return None, 'file does not exist'

    img = None
    for attempt in range(1, retries + 1):
        img = cv2.imread(image_path)
        if img is not None and img.size > 0:
            return img, None
        if attempt < retries:
            time.sleep(delay_s * attempt)

    return None, f'cv2.imread returned None/empty after {retries} retries'
