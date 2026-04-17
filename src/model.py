import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

__all__ = ['predict_contact_batch']

try:
    from .utils import crop_tangent_square
except ImportError:
    from utils import crop_tangent_square


def predict_contact_batch(f_bond_frame, I, model, plot_raw=False, batch_size=32):


    """
    Crop contact regions, normalise, and run batched ResNet inference.

    Returns:
        preds       – (N, num_classes) float32 softmax probabilities
        batch_crops – (N, 128, 128, 3) uint8 raw crops
    """
            # ImageNet normalisation constants (must match training transforms)
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    device = next(model.parameters()).device
    n = len(f_bond_frame)
    batch_crops = np.empty((n, 128, 128, 3), dtype=np.uint8)

    for idx, row in enumerate(f_bond_frame.itertuples(index=False)):
        angle    = np.arctan2(row.yj - row.yi, row.xj - row.xi)
        ri       = row.ri
        x1       = int(row.xi + ri * np.cos(angle))
        y1       = int(row.yi + ri * np.sin(angle))
        cropped  = crop_tangent_square(I, (x1, y1), angle, int(1.2 * ri))
        cropped  = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
        cropped  = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
        batch_crops[idx] = cropped

        if plot_raw:
            plt.figure(figsize=(4, 4))
            plt.imshow(cropped)
            plt.show()

    imgs = torch.from_numpy(batch_crops).float() / 255.0   # (N, H, W, C)
    imgs = imgs.permute(0, 3, 1, 2)                        # (N, C, H, W)
    imgs = (imgs - _IMAGENET_MEAN) / _IMAGENET_STD

    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            logits = model(imgs[start:start + batch_size].to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(all_probs, axis=0), batch_crops



