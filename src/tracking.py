"""
tracking_utils.py
-----------------
Trajectory / particle-tracking helpers.

Functions
---------
max_num                 – find the largest frame index in a folder
interpolate_pos_angle   – linearly fill in missing particle positions & angles
draw_particle_orientation – overlay orientation lines on an image
"""

import numpy as np
import pandas as pd
import cv2

__all__ = [
    'interpolate_pos_angle',
    'fit_refined_centroid',
    'refine_frame_centers',
    'fill_temporal_single_frame_gaps',
]

def interpolate_pos_angle(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate x/y/angle for frames where a particle was not detected."""
    new_rows = []
    total_missing = 0

    for pid, particle_data in df.groupby('particle'):
        particle_data = particle_data.set_index('frame').sort_index()
        frames = particle_data.index.to_numpy()
        if len(frames) == 0:
            continue

        missing = sorted(set(range(frames[0], frames[-1] + 1)) - set(frames))
        if not missing:
            continue

        total_missing += len(missing)
        placeholders = pd.DataFrame(index=missing, columns=particle_data.columns)
        placeholders['particle'] = pid
        placeholders.index.name = 'frame'

        combined = pd.concat([particle_data, placeholders]).sort_index()
        combined[['x', 'y', 'angle', 'boundary']] = (
            combined[['x', 'y', 'angle', 'boundary']]
            .astype(float)
            .interpolate(method='linear', limit_direction='both')
        )
        for col in ['rpx']:
            combined[col] = combined[col].ffill().bfill()

        combined = combined.reset_index()
        new_rows.append(combined[combined['frame'].isin(missing)])

    if new_rows:
        df = pd.concat([df, *new_rows], ignore_index=True)

    df = df.sort_values(['particle', 'frame']).reset_index(drop=True)
    print(f'Interpolated {total_missing} missing positions.')
    return df


def fit_refined_centroid(patch: np.ndarray):
    """Compute intensity-weighted centroid of a patch for sub-pixel refinement.

    Parameters
    ----------
    patch : 2-D float array (DoG response around a particle centre)

    Returns
    -------
    cx  : float – refined x centre relative to patch origin
    cy  : float – refined y centre relative to patch origin
    rss : float – variance of the weight map (quality indicator)
    """
    h, w = patch.shape
    p_min = patch.min()
    safe = patch - p_min
    weights = safe ** 2  # sharpen contrast

    sum_w = np.sum(weights)
    if sum_w == 0:
        return w / 2, h / 2, 0.0

    yy, xx = np.mgrid[0:h, 0:w]
    cx = np.sum(xx * weights) / sum_w
    cy = np.sum(yy * weights) / sum_w
    rss = float(np.var(weights))

    return cx, cy, rss


def refine_frame_centers(frame_data, I_frame, kernels: dict, roi: tuple, max_shift: int = 10):
    """Apply DoG convolution + centroid refinement to all particles in one frame.

    Parameters
    ----------
    frame_data : DataFrame slice for the current frame (columns: x, y, rpx, ...)
    I_frame    : full BGR image for this frame (already camera-aligned)
    kernels    : dict mapping rpx value → DoG kernel, e.g. {46: big_kernel, 37: small_kernel}
    roi        : (y_min, y_max, x_min, x_max) crop applied to the image
    max_shift  : maximum allowed centroid shift in pixels before falling back

    Returns
    -------
    List of Series (one per particle) with refined x, y and refine_shift added.
    """
    Ig = I_frame[roi[0]:roi[1], roi[2]:roi[3], 0].astype(np.float32)
    Ig_norm = (Ig - Ig.min()) / (Ig.max() - Ig.min() + 1e-6)

    # Pre-compute one DoG response per kernel
    responses = {
        rpx: cv2.filter2D(Ig_norm, -1, k)
        for rpx, k in kernels.items()
    }
    # Clip negatives (background suppression)
    for rpx in responses:
        r = responses[rpx] - 100
        r[r < 0] = 0
        responses[rpx] = r

    refined = []
    for _, row in frame_data.iterrows():
        x_c, y_c = float(row['x']), float(row['y'])
        r = float(row['rpx'])

        dog_response = responses.get(r, responses[min(responses, key=lambda k: abs(k - r))])

        roi_half = int(max(1, round(0.25 * r)))
        x0 = int(max(0, round(x_c - roi_half)))
        y0 = int(max(0, round(y_c - roi_half)))
        x1 = int(min(dog_response.shape[1], round(x_c + roi_half)))
        y1 = int(min(dog_response.shape[0], round(y_c + roi_half)))

        patch_dog = dog_response[y0:y1, x0:x1].copy()
        cx_patch, cy_patch, _ = fit_refined_centroid(patch_dog)

        x_refined = cx_patch + x0
        y_refined = cy_patch + y0
        shift_dist = np.hypot(x_refined - x_c, y_refined - y_c)

        if shift_dist > max_shift or not np.isfinite(x_refined) or not np.isfinite(y_refined):
            x_refined, y_refined, shift_dist = x_c, y_c, 0.0

        row_out = row.copy()
        row_out['x'] = float(x_refined)
        row_out['y'] = float(y_refined)
        row_out['refine_shift'] = float(shift_dist)
        refined.append(row_out)

    return refined


def fill_temporal_single_frame_gaps(df):
    """Promote contact at t when same pair is contact at t-1 and t+1.

    Applies only to rows that already passed stage-1 geometric candidate selection.
    Requires consecutive frames for the 1-0-1 bridge.
    """
    out = df.copy()
    if out.empty:
        return out

    pair_lo = np.minimum(out['i'].to_numpy(), out['j'].to_numpy())
    pair_hi = np.maximum(out['i'].to_numpy(), out['j'].to_numpy())
    out['_pair_lo'] = pair_lo
    out['_pair_hi'] = pair_hi

    out = out.sort_values(['_pair_lo', '_pair_hi', 'frame']).reset_index(drop=True)

    for _, idx in out.groupby(['_pair_lo', '_pair_hi'], sort=False).groups.items():
        grp = out.loc[idx]
        if len(grp) < 3:
            continue

        frames = grp['frame'].to_numpy(dtype=int)
        contact = grp['contact'].to_numpy(dtype=np.int8)

        bridge_mask = (
            (contact[1:-1] == 0)
            & (contact[:-2] == 1)
            & (contact[2:] == 1)
            & ((frames[1:-1] - frames[:-2]) == 1)
            & ((frames[2:] - frames[1:-1]) == 1)
        )

        if np.any(bridge_mask):
            contact[1:-1][bridge_mask] = 1
            out.loc[idx, 'contact'] = contact

    return out.drop(columns=['_pair_lo', '_pair_hi'])
