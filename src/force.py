
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from .utils import get_disk_img, smooth_image

__all__ = [
    'get_model',
    'StressSolve_residue_torch',
    'synth_img_pytorch_residue',
    'fit_disk_residue',
    'fit_one_particle_cpu',
    'fit_one_particle_gpu',
    'symmetrize_forces',
]


def symmetrize_forces(F_bond_out: pd.DataFrame):
    """Build ij/ji reciprocity metrics and symmetrize using lower fitLoss.

    Parameters
    ----------
    F_bond_out : pd.DataFrame
        Fitted contact table with at least:
        ['frame', 'i', 'j', 'force', 'alpha', 'fitLoss'].

    Returns
    -------
    F_compare : pd.DataFrame
        Reciprocal-pair comparison table with force/alpha difference metrics.
    F_bond_corrected : pd.DataFrame
        Symmetrized table where each reciprocal pair uses the lower-loss side.
    stats : dict
        Summary counts for quick reporting.
    """
    required_cols = {'frame', 'i', 'j', 'force', 'alpha', 'fitLoss'}
    missing = required_cols.difference(F_bond_out.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    # Keep one row per directed contact before reciprocal matching.
    F_unique = F_bond_out.copy()
    F_unique.drop_duplicates(subset=['frame', 'i', 'j'], keep='first', inplace=True)

    F_swap = F_unique[['frame', 'i', 'j', 'force', 'alpha', 'fitLoss']].copy()
    F_swap.columns = ['frame', 'j', 'i', 'force_ji', 'alpha_ji', 'fitLoss_ji']
    F_compare = F_unique.merge(F_swap, on=['frame', 'i', 'j'], how='inner')

    F_compare['force_diff'] = F_compare['force'] - F_compare['force_ji']
    F_compare['force_diff_normalized'] = F_compare['force_diff'] / (F_compare['force_ji'] + 1e-10)
    F_compare['alpha_diff'] = np.abs(F_compare['alpha']) - np.abs(F_compare['alpha_ji'])
    F_compare['alpha_diff_norm'] = np.arctan2(
        np.sin(F_compare['alpha_diff']),
        np.cos(F_compare['alpha_diff']),
    )
    F_compare['alpha_opposite'] = np.abs(np.abs(F_compare['alpha_diff_norm']) - np.pi) < 0.1

    F_bond_corrected = F_unique.copy()
    F_bond_corrected['force'] = np.abs(F_bond_corrected['force'])

    best_values = {}
    for _, row in F_compare.iterrows():
        frame = int(row['frame'])
        i = int(row['i'])
        j = int(row['j'])

        if row['fitLoss'] <= row['fitLoss_ji']:
            best_force = np.abs(row['force'])
            best_alpha = row['alpha']
        else:
            best_force = np.abs(row['force_ji'])
            best_alpha = row['alpha_ji']

        best_values[(frame, i, j)] = (best_force, best_alpha)
        best_values[(frame, j, i)] = (best_force, best_alpha)

    for idx, row in F_bond_corrected.iterrows():
        key = (int(row['frame']), int(row['i']), int(row['j']))
        if key in best_values:
            F_bond_corrected.loc[idx, 'force'] = best_values[key][0]
            F_bond_corrected.loc[idx, 'alpha'] = best_values[key][1]

    stats = {
        'total_contacts': int(len(F_bond_corrected)),
        'reciprocal_pairs': int(len(best_values) // 2),
        'contacts_unchanged': int(len(F_bond_corrected) - len(best_values) // 2),
    }
    return F_compare, F_bond_corrected, stats

def get_model(device, output_dim=2):
    """
    Build a ResNet18-based regression model for contact force and angle prediction.

    Parameters
    ----------
    device     : torch.device
    output_dim : int   number of output values (default 2 -> [force, angle])
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, output_dim)
    )
    return model.to(device)


# Global coordinate mesh cache - avoids recomputing on every call
_mesh_cache = {}

# TorchScript handles the hot stress kernel on older GPUs where torch.compile is unavailable.
USE_JIT_STRESS = False
STRESS_SOLVER_JIT = None


def StressSolve_residue_torch(xxi: torch.Tensor, xxj: torch.Tensor, f: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, fsigma: float, rm: float, power: float, eps: float = 1e-10) -> torch.Tensor:
    """
    Vectorised photoelastic intensity for a disk with z contacts.

    xxi, xxj : [HW] pixel coordinate tensors (inside unit disk)
    f, alpha, beta : [z] contact force magnitude, opening angle, contact angle
    Returns [HW] intensity tensor.
    """
    device = xxi.device
    beta_adj = -beta + torch.pi / 2

    pioverfsigma = torch.pi / fsigma
    twooverpi = 2 / torch.pi

    xxi_exp = xxi.unsqueeze(0)
    xxj_exp = xxj.unsqueeze(0)
    f_exp = f.unsqueeze(1)
    alpha_exp = alpha.unsqueeze(1)
    beta_exp = beta_adj.unsqueeze(1)

    b = beta_exp + torch.pi / 2
    a = alpha_exp
    b2 = torch.where(a < 0, b + (torch.pi + 2 * a), b - (torch.pi - 2 * a))

    x1 = rm * torch.sin(b)
    y1 = rm * torch.cos(b)
    x2 = rm * torch.sin(b2)
    y2 = rm * torch.cos(b2)

    ch0 = x2 - x1
    ch1 = y2 - y1
    chn = torch.sqrt(ch0**2 + ch1**2 + eps)
    ch0 = ch0 / chn
    ch1 = ch1 / chn

    r10 = xxi_exp - x1
    r11 = -xxj_exp - y1
    r1n = torch.sqrt(r10**2 + r11**2 + eps)

    costh1 = torch.clamp((r10 * ch0 + r11 * ch1) / r1n, -1 + eps, 1 - eps)
    signth = torch.where(r11 * ch0 > r10 * ch1, 1.0, -1.0)
    th1 = signth * torch.acos(costh1)

    s1 = -(f_exp * twooverpi) / r1n * costh1
    th = th1 - beta_exp - alpha_exp

    sigmaxx = torch.sum(s1 * torch.sin(th)**2, dim=0)
    sigmayy = torch.sum(s1 * torch.cos(th)**2, dim=0)
    sigmaxy = torch.sum(0.5 * s1 * torch.sin(2 * th), dim=0)

    # Residual stress
    R = torch.sqrt(xxi**2 + xxj**2 + eps)
    K_res = -fsigma / (torch.pi * power * rm**power) * torch.arcsin(torch.tensor(1.0, device=device) ** 0.5)
    s_r_res = K_res * (rm**power - R**power)
    s_t_res = K_res * (rm**power - (power + 1) * R**power)
    Theta = torch.atan2(xxi, xxj)

    cos2, sin2 = torch.cos(Theta)**2, torch.sin(Theta)**2
    sincos = torch.sin(Theta) * torch.cos(Theta)

    sigmaxx_tot = sigmaxx + s_r_res * cos2 + s_t_res * sin2
    sigmayy_tot = sigmayy + s_r_res * sin2 + s_t_res * cos2
    sigmaxy_tot = sigmaxy + (s_r_res - s_t_res) * sincos

    aa = torch.sqrt((sigmaxx_tot - sigmayy_tot)**2 + 4 * sigmaxy_tot**2)
    result = torch.sin(pioverfsigma * aa)**2
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# Try TorchScript for stress kernel; if script fails, stay in eager mode.
try:
    STRESS_SOLVER_JIT = torch.jit.script(StressSolve_residue_torch)
    USE_JIT_STRESS = True
    print('TorchScript enabled for StressSolve_residue_torch.')
except Exception as e:
    USE_JIT_STRESS = False
    STRESS_SOLVER_JIT = None
    print(f'TorchScript disabled; using eager stress solver. Reason: {e}')


def synth_img_pytorch_residue(fsigma, rm, px, f, alpha, beta, device='cuda'):
    """Generate a synthetic photoelastic disk image on a px x px grid."""
    cache_key = (px, rm, device)
    if cache_key not in _mesh_cache:
        x = torch.linspace(-rm, rm, px, device=device)
        xxi, xxj = torch.meshgrid(x, x, indexing='ij')
        xxi_flat, xxj_flat = xxi.reshape(-1), xxj.reshape(-1)
        mask = xxi_flat**2 + xxj_flat**2 <= rm**2
        _mesh_cache[cache_key] = {
            'xxi_flat': xxi_flat,
            'xxj_flat': xxj_flat,
            'mask': mask,
            'template': torch.zeros(px * px, device=device)
        }
    c = _mesh_cache[cache_key]

    stress_fn = STRESS_SOLVER_JIT if USE_JIT_STRESS and STRESS_SOLVER_JIT is not None else StressSolve_residue_torch
    intensity = stress_fn(
        c['xxi_flat'][c['mask']], c['xxj_flat'][c['mask']],
        f, alpha, beta, fsigma, rm, power=10
    )
    img = c['template'].clone()
    img[c['mask']] = intensity
    return img.reshape(px, px)


def fit_disk_residue(photo_img, fsigma, rm, px, f0, alpha0, beta,
                     lr, n_iter=1000, device='cuda', verbose=0,
                     tol=5e-4, patience=300):
    """
    Optimise (f0, alpha0) to minimise image reconstruction + equilibrium losses.

    Returns: f_fit, alpha_fit, final_loss, loss_history
    """
    f0 = f0.clone().detach().to(device).requires_grad_(True)
    alpha0 = alpha0.clone().detach().to(device).requires_grad_(True)
    beta = beta.to(device)

    optimizer = torch.optim.Adam([f0, alpha0], lr=lr)
    pi_half = torch.tensor(torch.pi / 2, device=device)
    torque_weight = torch.tensor(1e5, device=device)

    prev_loss = None
    patience_counter = 0
    loss_history = []

    for i in range(n_iter):
        optimizer.zero_grad()

        f0_pos = torch.abs(f0)
        synth = synth_img_pytorch_residue(fsigma, rm, px, f0_pos, alpha0, beta, device=device)

        image_loss = ((smooth_image(synth) - photo_img) ** 2).mean()
        angle_term = alpha0 - beta + pi_half
        torque_loss = torque_weight * torch.sum(torch.sin(alpha0) * rm * f0_pos) ** 2
        force_loss = torch.sum(torch.cos(angle_term) * f0_pos) ** 2 + \
                     torch.sum(torch.sin(angle_term) * f0_pos) ** 2
        loss = image_loss + torque_loss + force_loss

        if verbose and i % 10 == 0:
            print(f"Iter {i}: loss={loss.item():.4f}  image={image_loss.item():.4f}  "
                  f"torque={torque_loss.item():.4f}  force={force_loss.item():.4f}")

        if torch.isnan(loss) or torch.isinf(loss):
            print('Loss diverged - stopping.')
            break

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if prev_loss is not None:
            patience_counter = patience_counter + 1 if prev_loss - current_loss < tol else 0
            if patience_counter >= patience:
                break
        prev_loss = current_loss

    return f0.detach().cpu().numpy(), alpha0.detach().cpu().numpy(), loss.detach(), loss_history


def fit_one_particle_cpu(
    particle_id,
    pdata_particle,
    img,
    fsigma,
    *,
    device,
    lr,
    n_iter=1000,
    tol=5e-4,
    patience=300,
    verbose=0,
    do_plot=False,
):
    """Fit one particle on CPU and return ``(particle_id, pdata_out_df)``."""
    if img is None:
        raise ValueError('fit_one_particle_cpu received img=None (image decode/read failed upstream).')

    z = len(pdata_particle)
    if z <= 1:
        return None

    pdata = pdata_particle.reset_index(drop=True).copy()

    gray_img = get_disk_img(pdata, img)
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.astype(np.float32) / 255.0 if gray_img.max() > 1 else gray_img.astype(np.float32)

    rm = pdata.iloc[0]['ri'] / 37 * 6 / 1000
    img_size = gray_img.shape[0]

    alphas = pdata['angle_pred'].to_numpy().copy()
    betas = pdata['beta'].to_numpy().copy()
    forces = pdata['force_pred'].to_numpy().copy()

    if verbose:
        print(f'Particle id: {particle_id}, Contact number z={z}')
        print(f'Initial forces: {forces}')
        print(f'Initial alphas: {alphas}')

    f0_cpu = torch.tensor(forces.tolist(), dtype=torch.float32, device=device)
    alpha0_cpu = torch.tensor(alphas.tolist(), dtype=torch.float32, device=device)
    betas_cpu = torch.tensor(betas.tolist(), dtype=torch.float32, device=device)
    gray_img_cpu = torch.tensor(gray_img, dtype=torch.float32, device=device)

    # Warm mesh cache to reduce first-iteration latency.
    _ = synth_img_pytorch_residue(fsigma, rm, img_size, f0_cpu, alpha0_cpu, betas_cpu, device=device)

    f_fit, alpha_fit, fitted_loss, _ = fit_disk_residue(
        gray_img_cpu,
        fsigma,
        rm,
        img_size,
        f0_cpu,
        alpha0_cpu,
        betas_cpu,
        verbose=verbose,
        tol=tol,
        patience=patience,
        lr=lr,
        n_iter=n_iter,
        device=device,
    )

    if do_plot and fitted_loss.cpu().numpy() > 0:
        fit_im = synth_img_pytorch_residue(
            fsigma,
            rm,
            img_size,
            torch.tensor(f_fit, dtype=torch.float32, device=device),
            torch.tensor(alpha_fit, dtype=torch.float32, device=device),
            betas_cpu,
            device=device,
        )
        guess_im = synth_img_pytorch_residue(fsigma, rm, img_size, f0_cpu, alpha0_cpu, betas_cpu, device=device)

        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(gray_img_cpu.cpu().numpy(), cmap='gray', vmax=1)
        plt.title(f'id = {particle_id} \n exp', fontsize=10)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(smooth_image(guess_im, kernel_size=3, sigma=1.0).cpu().numpy(), cmap='gray', vmax=1)
        plt.title('guess', fontsize=10)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(smooth_image(fit_im, kernel_size=3, sigma=1.0).cpu().numpy(), cmap='gray', vmax=1)
        plt.title('fit', fontsize=10)
        plt.axis('off')
        plt.show()

    pdata_out = pdata.copy()
    pdata_out['force'] = f_fit
    pdata_out['alpha'] = alpha_fit
    pdata_out['fitLoss'] = fitted_loss.cpu().numpy()

    return particle_id, pdata_out


def fit_one_particle_gpu(
    particle_id,
    pdata_particle,
    img,
    fsigma,
    *,
    device,
    lr,
    n_iter=1000,
    tol=5e-4,
    patience=300,
    verbose=0,
    do_plot=False,
):
    """Fit one particle on GPU and return ``(particle_id, pdata_out_df)``.

    All previously implicit notebook dependencies are explicit parameters here.
    """
    if img is None:
        raise ValueError('fit_one_particle_gpu received img=None (image decode/read failed upstream).')

    z = len(pdata_particle)
    if z <= 1:
        return None

    pdata = pdata_particle.reset_index(drop=True).copy()

    gray_img = get_disk_img(pdata, img)
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.astype(np.float32) / 255.0 if gray_img.max() > 1 else gray_img.astype(np.float32)

    rm = pdata.iloc[0]['ri'] / 37 * 6 / 1000
    img_size = gray_img.shape[0]

    alphas = pdata['angle_pred'].to_numpy().copy()
    betas = pdata['beta'].to_numpy().copy()
    forces = pdata['force_pred'].to_numpy().copy()

    if verbose:
        print(f'Particle id: {particle_id}, Contact number z={z}')
        print(f'Initial forces: {forces}')
        print(f'Initial alphas: {alphas}')

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        f0_gpu = torch.tensor(forces.tolist(), dtype=torch.float32, device=device)
        alpha0_gpu = torch.tensor(alphas.tolist(), dtype=torch.float32, device=device)
        betas_gpu = torch.tensor(betas.tolist(), dtype=torch.float32, device=device)
        gray_img_gpu = torch.tensor(gray_img, dtype=torch.float32, device=device)

        _ = synth_img_pytorch_residue(fsigma, rm, img_size, f0_gpu, alpha0_gpu, betas_gpu, device=device)

        f_fit, alpha_fit, fitted_loss, _ = fit_disk_residue(
            gray_img_gpu,
            fsigma,
            rm,
            img_size,
            f0_gpu,
            alpha0_gpu,
            betas_gpu,
            verbose=verbose,
            tol=tol,
            patience=patience,
            lr=lr,
            n_iter=n_iter,
            device=device,
        )

        if do_plot and fitted_loss.cpu().numpy() > 0:
            fit_im = synth_img_pytorch_residue(
                fsigma,
                rm,
                img_size,
                torch.tensor(f_fit, dtype=torch.float32, device=device),
                torch.tensor(alpha_fit, dtype=torch.float32, device=device),
                betas_gpu,
                device=device,
            )
            guess_im = synth_img_pytorch_residue(fsigma, rm, img_size, f0_gpu, alpha0_gpu, betas_gpu, device=device)

    stream.synchronize()

    if do_plot and fitted_loss.cpu().numpy() > 0:
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(gray_img_gpu.cpu().numpy(), cmap='gray', vmax=1)
        plt.title(f'id = {particle_id} \n exp', fontsize=10)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(smooth_image(guess_im, kernel_size=3, sigma=1.0).cpu().numpy(), cmap='gray', vmax=1)
        plt.title('guess', fontsize=10)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(smooth_image(fit_im, kernel_size=3, sigma=1.0).cpu().numpy(), cmap='gray', vmax=1)
        plt.title('fit', fontsize=10)
        plt.axis('off')
        plt.show()

    pdata_out = pdata.copy()
    pdata_out['force'] = f_fit
    pdata_out['alpha'] = alpha_fit
    pdata_out['fitLoss'] = fitted_loss.cpu().numpy()

    return particle_id, pdata_out
