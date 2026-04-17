"""
orientation.py
--------------
Disk orientation / rotation-angle helpers.

Functions
---------
orientation_weighted_pca    – intensity-weighted PCA for sub-pixel orientation
compute_frame_orientations  – run orientation PCA for every particle in one frame image
compute_continuous_angles   – unwrap per-particle PCA angles across frames
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

__all__ = [
    'orientation_weighted_pca',
    'compute_continuous_angles',
    'compute_frame_orientations',
]


def orientation_weighted_pca(img_gray: np.ndarray):
    """Intensity-weighted PCA for sub-pixel orientation precision.

    Parameters
    ----------
    img_gray : 2-D float or uint8 array (single channel)

    Returns
    -------
    direction : ndarray, shape (2,)
    R2        : float
    """
    img = gaussian_filter(img_gray.astype(float), sigma=1.0)

    threshold = np.mean(img) + 0.5 * np.std(img)
    mask = img > threshold

    y, x = np.where(mask)
    weights = img[mask]

    if len(x) < 3:
        return np.array([1.0, 0.0]), 0.0

    xc = np.average(x, weights=weights)
    yc = np.average(y, weights=weights)

    x_c = x - xc
    y_c = y - yc

    cov_xx = np.average(x_c * x_c, weights=weights)
    cov_yy = np.average(y_c * y_c, weights=weights)
    cov_xy = np.average(x_c * y_c, weights=weights)

    cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    direction = eigenvectors[:, 1]
    denom = eigenvalues[0] ** 2 + eigenvalues[1] ** 2
    R2 = eigenvalues[1] ** 2 / denom if denom > 0 else 0.0

    return direction, R2


def compute_continuous_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Accumulate unwrapped rotation angles across frames for each particle.

    Handles the π-ambiguity of PCA orientation vectors: at each step the branch
    (raw_angle or raw_angle ± π) closest to the previous accumulated direction is
    chosen.  Operates on numpy arrays to avoid per-row pandas overhead.

    Expects columns ``dir_x``, ``dir_y``, ``particle``, ``frame`` in *df*.
    Writes results to the ``angle`` column and returns the modified DataFrame.
    """
    df = df.sort_values(['particle', 'frame']).copy()
    df['angle'] = np.nan

    raw = np.arctan2(
        df['dir_y'].to_numpy(dtype=float),
        df['dir_x'].to_numpy(dtype=float),
    )
    pids = df['particle'].to_numpy()
    angle_out = np.full(len(df), np.nan)

    prev_raw: dict = {}
    prev_acc: dict = {}

    for k in range(len(df)):
        r = raw[k]
        if np.isnan(r):
            continue
        pid = pids[k]
        if pid not in prev_acc:
            prev_acc[pid] = r
            prev_raw[pid] = r
        else:
            pr   = prev_raw[pid]
            opt2 = r + np.pi if r < 0 else r - np.pi
            d1   = np.angle(np.exp(1j * (r    - pr)))
            d2   = np.angle(np.exp(1j * (opt2 - pr)))
            prev_acc[pid] += d1 if abs(d1) < abs(d2) else d2
            prev_raw[pid]  = r
        angle_out[k] = prev_acc[pid]

    df['angle'] = angle_out
    return df


def compute_frame_orientations(f: pd.DataFrame, I_blue: np.ndarray,
                               roi_half_scale: float = 0.6) -> list:
    """Run weighted-PCA orientation on every particle in one frame.

    Parameters
    ----------
    f               : DataFrame slice for this frame (columns: x, y, rpx, ...)
    I_blue          : 2-D grayscale image (blue channel, already cropped to ROI)
    roi_half_scale  : fraction of rpx used as the half-size of the UV crop

    Returns
    -------
    List of (original_index, dir_x, dir_y, R2) tuples for rows with a valid patch.
    """
    import cv2
    records = []
    for idx, row in f.iterrows():
        yc, xc = row['y'], row['x']
        half = row['rpx'] * roi_half_scale
        y1, y2 = int(yc - half), int(yc + half)
        x1, x2 = int(xc - half), int(xc + half)
        uv_roi = I_blue[y1:y2, x1:x2]
        if uv_roi.size > 0 and uv_roi.shape[0] > 0 and uv_roi.shape[1] > 0:
            uv_roi = cv2.GaussianBlur(uv_roi, (5, 5), 1)
            direction, R2 = orientation_weighted_pca(uv_roi)
            records.append((idx, direction[0], direction[1], R2))
    return records
