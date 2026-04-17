
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from .force import synth_img_pytorch_residue
from .utils import smooth_image, crop_circle_with_mask_float

__all__ = ['draw_particle_orientation', 'plot_contacts', 'build_particle_synth_output_img']


def build_particle_synth_output_img(f: pd.DataFrame, orig_img: np.ndarray, fsigma: float, device='cuda') -> np.ndarray:
    """Build the synthesized field by accumulating per-particle fitted patches.

    Parameters
    ----------
    f : pd.DataFrame
        Corrected contact table for a single frame.
    orig_img : np.ndarray
        Original cropped PE image used as canvas size reference.
    fsigma : float
        Photoelastic constant.
    device : str or torch.device
        Torch device for synthesis.

    Returns
    -------
    np.ndarray
        Accumulated synthesized image (`output_img`) with the same shape as orig_img.
    """
    output_img = np.zeros_like(orig_img, dtype=np.float32)
    h, w = orig_img.shape[:2]
    is_rgb = len(orig_img.shape) == 3 and orig_img.shape[2] == 3

    with torch.inference_mode():
        for _, entry in f.groupby('i', sort=False):
            if len(entry) <= 1:
                continue

            x = int(np.around(entry['xi'].iloc[0]))
            y = int(np.around(entry['yi'].iloc[0]))
            r = int(entry['ri'].iloc[0])

            y1, y2 = max(0, y - r), min(h, y + r)
            x1, x2 = max(0, x - r), min(w, x + r)
            crop_h, crop_w = y2 - y1, x2 - x1
            if crop_h <= 0 or crop_w <= 0:
                continue

            rm = entry['ri'].iloc[0] / 37 * 6 / 1000
            px = crop_h
            f0 = torch.tensor(entry['force'].to_numpy(), dtype=torch.float32, device=device)
            alpha0 = torch.tensor(entry['alpha'].to_numpy(), dtype=torch.float32, device=device)
            beta = torch.tensor(entry['beta'].to_numpy(), dtype=torch.float32, device=device)

            fit_im = synth_img_pytorch_residue(fsigma, rm, px, f0, alpha0, beta, device=device)
            im_smoothed = smooth_image(fit_im, kernel_size=3, sigma=1.0)
            fit_img_np = im_smoothed.cpu().numpy()
            fit_img_cropped = crop_circle_with_mask_float(fit_img_np)
            fit_img_resized = cv2.resize(fit_img_cropped, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

            if is_rgb:
                fit_img_resized = np.stack([fit_img_resized] * 3, axis=-1)

            output_img[y1:y2, x1:x2] += fit_img_resized

    return output_img

def draw_particle_orientation(
    img: np.ndarray,
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    angle_col: str = 'angle',
    linewidth: int = 1,
    linecolor: str = 'r',
    circle_color: str = 'r',
    show: bool = False,
) -> np.ndarray:
    """Overlay orientation lines and particle circles on *img*.

    Returns a BGR uint8 array.
    """
    img_draw = img.copy()
    if img_draw.ndim == 2:
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)

    df_plot = df[df[angle_col].notnull()].copy()
    frame = None

    for _, row in df_plot.iterrows():
        xc    = float(row[x_col])
        yc    = float(row[y_col])
        theta = float(row[angle_col])
        frame = int(row['frame']) if 'frame' in row else None
        half  = float(row['rpx'])

        dx = np.cos(theta) * half
        dy = np.sin(theta) * half

        pt1 = (int(round(xc - dx)), int(round(yc - dy)))
        pt2 = (int(round(xc + dx)), int(round(yc + dy)))
        lc = (np.array(mcolors.to_rgb(linecolor)) * 255).tolist()
        cc = (np.array(mcolors.to_rgb(circle_color)) * 255).tolist()
        cv2.line(img_draw, pt1, pt2, lc, thickness=linewidth, lineType=cv2.LINE_AA)
        cv2.circle(
            img_draw,
            (int(round(xc)), int(round(yc))),
            int(row['rpx']),
            cc,
            thickness=linewidth,
            lineType=cv2.LINE_AA,
        )

    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        title = 'Orientation overlay'
        if frame is not None:
            title += f' (frame {frame})'
        plt.title(title)
        plt.show()

    return img_draw

def plot_contacts(I, f, F_out, boundary_pid):
    """Render image with particle circles and contact lines. Returns RGB array."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(I, cmap='gray')
    for _, row in f.iterrows():
        ax.add_patch(plt.Circle((row['x'], row['y']), row['rpx'],
                                color='green', fill=False, linewidth=1))
    for _, row in f[f.particle.isin(boundary_pid)].iterrows():
        ax.add_patch(plt.Circle((row['x'], row['y']), row['rpx'],
                                color='red', fill=False, linewidth=1))
    for _, row in F_out.iterrows():
        ax.plot([row['xi'], row['xj']], [row['yi'], row['yj']],
                color='cyan', linestyle='--', linewidth=5, alpha=0.5)
    for _, row in F_out[F_out.singular > 0].iterrows():
        ax.plot([row['xi'], row['xj']], [row['yi'], row['yj']],
                color='magenta', linestyle='--', linewidth=2, alpha=0.4)
    ax.axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[..., :3].copy()
    plt.close(fig)
    return img