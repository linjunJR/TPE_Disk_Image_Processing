#!/usr/bin/env python
"""
TPE Image Processing Pipeline — batch script
=============================================
Runs Step 1 (disk tracking via StarDist) and Step 2 (contact detection via ResNet)
for a given experiment folder.

Usage
-----
    python run_pipeline.py --img-dir N:\\PROJ_TPE\\TPE_20260429A03_... [options]

Required
--------
    --img-dir       Full experiment directory containing image frames

Optional
--------
    --pkl-dir       Directory for trajectory .pkl       (default: M:\\Archive\\Proj_TPE\\Disk_traj_files)
    --bond-dir      Directory for contact-bond .pkl     (default: M:\\Archive\\Proj_TPE\\Contact_bond_files)
    --roi           y_min y_max x_min x_max             (default: 250 1200 0 2000)
    --d-tol         Neighbour distance tolerance [px]   (default: 10)
    --verbose       Save diagnostic figures next to output .pkl files
    --skip-tracking Skip Step 1; load existing trajectory .pkl
    --skip-contact  Skip Step 2; only run tracking

Direct run (no CLI flags)
--------------------------
    1) Set DIRECT_RUN_IMG_DIR below.
    2) Click "Run Python File" in VS Code.

Batch example (PowerShell)
--------------------------
    $dirs = @("N:\\PROJ_TPE\\TPE_20260429A01_...", "N:\\PROJ_TPE\\TPE_20260429A03_...")
    foreach ($d in $dirs) {
        python run_pipeline.py --img-dir $d --verbose
    }
"""

import argparse
import gc
import os
import sys
import cv2
import numpy as np
import pandas as pd


# Direct-run defaults for VS Code "Run Python File".
# If the script is run with no CLI args, these settings are used.
# Set to a single string OR a list of strings for batch processing.
DIRECT_RUN_IMG_DIR = [
    "N:\PROJ_TPE\TPE_20260506A_N=262x2_7SpeedSweep_strain=0.5_5e2FramesEach\speed1",
    "N:\PROJ_TPE\TPE_20260506A_N=262x2_7SpeedSweep_strain=0.5_5e2FramesEach\speed2",
    "N:\PROJ_TPE\TPE_20260506A_N=262x2_7SpeedSweep_strain=0.5_5e2FramesEach\speed3",
    "N:\PROJ_TPE\TPE_20260506A_N=262x2_7SpeedSweep_strain=0.5_5e2FramesEach\speed4",
    "N:\PROJ_TPE\TPE_20260506A_N=262x2_7SpeedSweep_strain=0.5_5e2FramesEach\speed5",
    "N:\PROJ_TPE\TPE_20260506A_N=262x2_7SpeedSweep_strain=0.5_5e2FramesEach\speed6",
]
DIRECT_RUN_VERBOSE = 1
DIRECT_RUN_SKIP_TRACKING = 0
DIRECT_RUN_SKIP_CONTACT = 1

# ── default paths and params ──────────────────────────────────────────────────────────
DEFAULT_PKL_DIR  = r'M:\Archive\Proj_TPE\Disk_traj_files'
DEFAULT_BOND_DIR = r'M:\Archive\Proj_TPE\Contact_bond_files'
DEFAULT_ROI      = (250, 1200, 0, 2000)
DEFAULT_D_TOL    = 10
DIRECT_RUN_PKL_DIR = DEFAULT_PKL_DIR
DIRECT_RUN_BOND_DIR = DEFAULT_BOND_DIR
DIRECT_RUN_ROI = list(DEFAULT_ROI)
DIRECT_RUN_D_TOL = DEFAULT_D_TOL

# ── camera calibration matrix — UPDATE HERE when recalibrating ─────────────
# Calibrated: 2026-05-06
# To recalibrate: replace the 9 numbers below with the new H matrix values.
CALIB_DATE = '2026-05-06'
DEFAULT_CALIB_H = np.array([
    [ 1.00709163e+00,  2.01940780e-02, -1.42063713e+00],
    [-1.81984493e-02,  1.01468029e+00, -3.83163172e+01],
    [-3.17462188e-06,  4.07314753e-07,  1.00000000e+00],
])


# ── CLI ───────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description='TPE disk-tracking + contact-detection pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--img-dir', default=DIRECT_RUN_IMG_DIR,
                   help='Full experiment directory containing input frames.')
    p.add_argument('--pkl-dir',  default=DIRECT_RUN_PKL_DIR)
    p.add_argument('--bond-dir', default=DIRECT_RUN_BOND_DIR)
    p.add_argument('--roi', nargs=4, type=int, default=DIRECT_RUN_ROI,
                   metavar=('Y_MIN', 'Y_MAX', 'X_MIN', 'X_MAX'))
    p.add_argument('--d-tol', type=int, default=DIRECT_RUN_D_TOL)
    p.add_argument('--verbose', action='store_true', default=DIRECT_RUN_VERBOSE,
                   help='Save diagnostic figures alongside output .pkl files.')
    p.add_argument('--skip-tracking', action='store_true', default=DIRECT_RUN_SKIP_TRACKING,
                   help='Skip Step 1; requires trajectory .pkl to already exist.')
    p.add_argument('--skip-contact', action='store_true', default=DIRECT_RUN_SKIP_CONTACT,
                   help='Skip Step 2; only run tracking.')
    return p


def build_direct_run_args():
    return argparse.Namespace(
        img_dir=DIRECT_RUN_IMG_DIR,
        pkl_dir=DIRECT_RUN_PKL_DIR,
        bond_dir=DIRECT_RUN_BOND_DIR,
        roi=DIRECT_RUN_ROI,
        d_tol=DIRECT_RUN_D_TOL,
        verbose=DIRECT_RUN_VERBOSE,
        skip_tracking=DIRECT_RUN_SKIP_TRACKING,
        skip_contact=DIRECT_RUN_SKIP_CONTACT,
    )


def resolve_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        print('[mode] CLI override mode.')
        return build_parser().parse_args(argv)
    print('[mode] direct-run defaults (no CLI args).')
    return build_direct_run_args()


# ── camera calibration ────────────────────────────────────────────────────
def make_camera_align():
    H = DEFAULT_CALIB_H
    print(f'[calib] using DEFAULT_CALIB_H (calibrated {CALIB_DATE}).')

    def camera_align(I):
        height, width = I.shape[:2]
        return cv2.warpPerspective(I, H, (width, height))

    return camera_align


# ── Step 1: disk tracking ─────────────────────────────────────────────────
def run_tracking(args, camera_align, diag_dir, src):
    import trackpy as tp
    from csbdeep.utils import normalize
    from skimage import measure
    from stardist.models import StarDist2D
    import tensorflow as tf

    roi = tuple(args.roi)
    exp_dir = args.img_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_parent = os.path.dirname(os.path.normpath(exp_dir))
    exp_folder = os.path.basename(os.path.normpath(exp_dir))

    max_frame_green = int(src.max_num(exp_parent, exp_folder, 'green_'))
    print(f'[tracking] {max_frame_green} green frames found.')

    # load StarDist
    model_dir = os.path.join(script_dir, 'models')
    model = StarDist2D(None, name='stardist_09221229', basedir=model_dir)

    # detect
    axis_norm = (0, 1)
    records = []
    for frame in range(1, max_frame_green + 1):
        path = os.path.join(exp_dir, f'green_{frame}.png')
        I = cv2.imread(path)
        if I is None:
            print(f'\n  skipping frame {frame} (not found)')
            continue
        I = camera_align(cv2.flip(I, 1))
        sys.stdout.write(f'\r[tracking] detecting — frame {frame}/{max_frame_green}')
        sys.stdout.flush()
        Ig = I[roi[0]:roi[1], roi[2]:roi[3], 1]
        X = normalize(Ig, 1, 99.8, axis=axis_norm)
        mask, detail = model.predict_instances(
            X, n_tiles=model._guess_n_tiles(X), show_tile_progress=False
        )
        for region in measure.regionprops(mask):
            y, x = region.centroid
            records.append({'frame': frame, 'x': x, 'y': y,
                             'area': region.area, 'ecc': region.eccentricity})
        del I, Ig, X, mask, detail
        if frame % 100 == 0:
            gc.collect()
            tf.keras.backend.clear_session()
    print()

    df = pd.DataFrame.from_records(records)

    # filter
    df = df[(df.x < 1860) & (df.x > 100) & (df.y > 50)]
    df = df[df.ecc < 0.6].copy()
    df['rpx'] = 46
    df.loc[df['area'] < 6000, 'rpx'] = 37

    # link
    F_linked = tp.link(df, search_range=30, memory=10)
    F_linked['boundary'] = (
        (F_linked.x < 200) | (F_linked.x > 1786) |
        (F_linked.y < 120) | (F_linked.y > 820)
    )
    F_linked['rpx'] = F_linked.groupby('particle')['rpx'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x
    )

    # rotation angles
    F_linked['dir_x'] = np.nan
    F_linked['dir_y'] = np.nan
    grouped = F_linked.groupby('frame')
    records_rot = []
    for frame in sorted(F_linked['frame'].unique()):
        sys.stdout.write(f'\r[tracking] orientations — frame {frame}')
        sys.stdout.flush()
        path = os.path.join(exp_dir, f'blue_{frame}.png')
        raw = cv2.imread(path)
        if raw is None:
            continue
        I = camera_align(cv2.flip(raw, 1))
        I = I[roi[0]:roi[1], roi[2]:roi[3], 0]
        records_rot.extend(src.compute_frame_orientations(grouped.get_group(frame), I))
    print()
    if records_rot:
        rot_df = (pd.DataFrame(records_rot, columns=['_idx', 'dir_x', 'dir_y', 'angle_R2'])
                  .set_index('_idx'))
        F_linked.update(rot_df)
    F_linked = src.compute_continuous_angles(F_linked)
    F_linked = src.interpolate_pos_angle(F_linked)
    F_linked.drop(['dir_x', 'dir_y'], axis=1, inplace=True)

    # DoG refinement
    print('[tracking] DoG refinement ...')
    kernels = {
        46: src.dog_kernel(49, delta=3, sigma=2),
        37: src.dog_kernel(40, delta=3, sigma=2),
    }
    refined_data = []
    max_frame_refine = int(F_linked['frame'].max())
    for frame in range(1, max_frame_refine + 1):
        frame_data = F_linked[F_linked['frame'] == frame]
        path = os.path.join(exp_dir, f'green_{frame}.png')
        I_frame = cv2.imread(path)
        if I_frame is None:
            refined_data.extend(frame_data.itertuples(index=False, name=None))
            continue
        I_frame = camera_align(cv2.flip(I_frame, 1))
        refined_data.extend(src.refine_frame_centers(frame_data, I_frame, kernels, roi))
    F_linked = pd.DataFrame(refined_data).reset_index(drop=True)
    print(f'[tracking] {len(F_linked)} detections after refinement.')

    # G² computation
    print('[tracking] computing per-particle G²...')
    sigma_frac_g2 = 0.1
    for frame in sorted(F_linked['frame'].unique()):
        sys.stdout.write(f'\r[tracking] G² — frame {frame}')
        sys.stdout.flush()
        image_path = os.path.join(exp_dir, f'bw_{frame}.png')
        img_g2 = cv2.imread(image_path)
        if img_g2 is None:
            continue
        img_g2 = img_g2[roi[0]:roi[1], roi[2]:roi[3]]
        frame_data_g2 = F_linked[F_linked['frame'] == frame]
        gray_g2 = cv2.cvtColor(img_g2, cv2.COLOR_BGR2GRAY)
        gray_g2 = cv2.GaussianBlur(gray_g2, (3, 3), 1).astype(np.float32)
        gray_g2 = gray_g2 / gray_g2.max()
        gray_g2 = src.subtract_gaussian_rings(gray_g2, frame_data_g2, sigma_frac=sigma_frac_g2)
        G2_map_run = src.compute_G2_map(gray_g2)
        h_g2, w_g2 = G2_map_run.shape
        for idx_g2, row_g2 in frame_data_g2.iterrows():
            x_g2 = int(np.around(float(row_g2['x'])))
            y_g2 = int(np.around(float(row_g2['y'])))
            r_g2 = int(row_g2['rpx'])
            y1_g2, y2_g2 = max(0, y_g2 - r_g2), min(h_g2, y_g2 + r_g2)
            x1_g2, x2_g2 = max(0, x_g2 - r_g2), min(w_g2, x_g2 + r_g2)
            if (y2_g2 - y1_g2) == 0 or (x2_g2 - x1_g2) == 0:
                F_linked.at[idx_g2, 'G2'] = np.nan
                continue
            G2_crop_run = src.crop_circle_with_mask_float(G2_map_run[y1_g2:y2_g2, x1_g2:x2_g2])
            F_linked.at[idx_g2, 'G2'] = float(np.sum(G2_crop_run[G2_crop_run > 0]))
    print('\n[tracking] G² computation complete.')

    # verbose diagnostics
    if args.verbose:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        all_frames_avail = F_linked['frame'].unique()
        sample_frames = np.random.choice(
            all_frames_avail,
            size=min(5, len(all_frames_avail)),
            replace=False,
        )

        # 5 tracking detection figures
        for frame_s in sample_frames:
            frame_s = int(frame_s)
            path_bw = os.path.join(exp_dir, f'bw_{frame_s}.png')
            test_img = cv2.imread(path_bw)
            if test_img is None:
                continue
            test_img = test_img[roi[0]:roi[1], roi[2]:roi[3], 0]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(test_img, cmap='gray')
            for _, row in F_linked[F_linked.frame == frame_s].iterrows():
                ax.add_patch(Circle((row.x, row.y), row.rpx,
                                    edgecolor='red', facecolor='none', linewidth=0.5))
            ax.axis('off')
            ax.set_title(f'Tracking detections — frame {frame_s}')
            out = os.path.join(diag_dir, f'tracking_frame{frame_s}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[tracking] diagnostic → {out}')

        # 5 G² overlay figures
        import matplotlib.patches as mpatches
        from matplotlib.colors import Normalize
        from matplotlib import cm as mpl_cm

        for frame_s in sample_frames:
            frame_s = int(frame_s)
            path_bw_g2 = os.path.join(exp_dir, f'bw_{frame_s}.png')
            img_diag = cv2.imread(path_bw_g2)
            if img_diag is None:
                continue
            img_diag = img_diag[roi[0]:roi[1], roi[2]:roi[3]]
            img_gray_diag = cv2.cvtColor(img_diag, cv2.COLOR_BGR2GRAY)
            fd_g2 = F_linked[F_linked.frame == frame_s].copy()
            f_vals_diag = fd_g2['G2'].to_numpy()
            if np.all(np.isnan(f_vals_diag)):
                continue
            vmin_d = np.nanpercentile(f_vals_diag, 1)
            vmax_d = np.nanpercentile(f_vals_diag, 99)
            cmap_d = plt.colormaps['jet']
            norm_d = Normalize(vmin=vmin_d, vmax=vmax_d)
            sm_d = mpl_cm.ScalarMappable(norm=norm_d, cmap=cmap_d)
            sm_d.set_array([])
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(img_gray_diag, cmap='gray', origin='upper')
            ax.set_aspect('equal')
            for _, row_d in fd_g2.iterrows():
                fc_d = cmap_d(norm_d(row_d.get('G2', np.nan)))
                ax.add_patch(mpatches.Circle(
                    (float(row_d['x']), float(row_d['y'])), float(row_d['rpx']),
                    edgecolor='none', facecolor=fc_d, alpha=0.75, zorder=2,
                ))
            fig.colorbar(sm_d, ax=ax, fraction=0.03, pad=0.04).set_label('G²')
            ax.set_title(f'G² overlay — frame {frame_s}')
            ax.axis('off')
            out = os.path.join(diag_dir, f'G2_overlay_frame{frame_s}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[tracking] G² diagnostic → {out}')

        # 5 orientation figures
        for frame_s in sample_frames:
            frame_s = int(frame_s)
            path_blue = os.path.join(exp_dir, f'blue_{frame_s}.png')
            I_blue = cv2.imread(path_blue)
            if I_blue is None:
                continue
            I_blue = camera_align(cv2.flip(I_blue, 1))
            I_blue = I_blue[roi[0]:roi[1], roi[2]:roi[3], 0]
            if I_blue.ndim == 2:
                I_blue = cv2.cvtColor(I_blue, cv2.COLOR_GRAY2BGR)
            res = src.draw_particle_orientation(
                I_blue.copy(),
                F_linked[F_linked.frame == frame_s],
                show=False, linecolor='c', linewidth=2,
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(res)
            ax.axis('off')
            ax.set_title(f'Orientation — frame {frame_s}')
            out = os.path.join(diag_dir, f'orientation_frame{frame_s}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[tracking] orientation diagnostic → {out}')

    return F_linked


# ── Step 2: contact detection ─────────────────────────────────────────────
def run_contact(args, camera_align, F_linked, diag_dir, src):
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models

    roi = tuple(args.roi)
    pe_dir = args.img_dir
    filetype = '.png'
    d_tol = args.d_tol
    script_dir = os.path.dirname(os.path.abspath(__file__))

    grouped_F = F_linked.groupby('frame')

    # build candidate bonds
    F_bond = []
    for frame in range(1, int(F_linked.frame.max()) + 1):
        sys.stdout.write(f'\r[contact] building bonds — frame {frame}')
        sys.stdout.flush()
        if frame not in grouped_F.groups:
            continue
        f = grouped_F.get_group(frame).reset_index(drop=True)
        boundary_pid = f.particle[f.boundary].to_numpy()
        F_bond_temp, *_ = src.get_all_bonds(f, boundary_pid, d_tol)
        F_bond_temp['frame'] = frame
        F_bond.append(F_bond_temp)
    print()
    F_bond = pd.concat(F_bond, ignore_index=True)
    print(f'[contact] {len(F_bond)} candidate bonds.')

    # load ResNet
    NUM_CLASSES = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(script_dir, 'models', 'ResNet18_contact_finetuned.pth')
    _backbone = tv_models.resnet18(weights=None)
    _backbone.fc = nn.Sequential(
        nn.Linear(_backbone.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, NUM_CLASSES),
    )
    _backbone.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model_nn = _backbone.to(device).eval()
    print(f'[contact] model loaded — running on {device}.')

    # classify
    pred_frames = []
    grouped_bond = F_bond.groupby('frame')
    for frame in range(1, int(F_linked.frame.max()) + 1):
        sys.stdout.write(f'\r[contact] classifying — frame {frame}')
        sys.stdout.flush()
        if frame not in grouped_F.groups or frame not in grouped_bond.groups:
            continue
        pe_path = os.path.join(pe_dir, f'bw_{frame}{filetype}')
        I = cv2.imread(pe_path, cv2.IMREAD_GRAYSCALE)
        if I is None:
            continue
        I = I[roi[0]:roi[1], roi[2]:roi[3]]
        f_bond_frame = grouped_bond.get_group(frame).copy()
        preds, _ = src.predict_contact_batch(f_bond_frame, I, model_nn,
                                             plot_raw=False, batch_size=32)
        f_bond_frame['contact'] = np.argmax(preds, axis=1)
        f_bond_frame['prob']    = np.max(preds, axis=1)
        pred_frames.append(f_bond_frame)
    print()

    F_pred = pd.concat(pred_frames, ignore_index=True)
    F_pred = src.fill_temporal_single_frame_gaps(F_pred)

    # post-process
    all_frames = []
    for frame in range(1, int(F_linked.frame.max()) + 1):
        if frame not in grouped_F.groups:
            continue
        f = grouped_F.get_group(frame).copy()
        boundary_pid = f.particle[f.boundary.astype(bool)].to_numpy()
        f_bond_frame = F_pred[F_pred.frame == frame].copy()
        f_bond_frame = f_bond_frame[f_bond_frame.contact > 0]
        if f_bond_frame.empty:
            continue
        f_bond_frame = src.process_singular_bonds(f_bond_frame, boundary_pid)
        f_bond_frame = src.duplicate_and_swap_bulk(f_bond_frame)
        all_frames.append(f_bond_frame)

    F_contact = pd.concat(all_frames, ignore_index=True).drop(columns=['contact'])
    print(f'[contact] {len(F_contact)} contact bonds.')

    # verbose diagnostics
    if args.verbose:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        all_frames_avail = F_contact['frame'].unique()
        sample_frames = np.random.choice(
            all_frames_avail,
            size=min(5, len(all_frames_avail)),
            replace=False,
        )

        for frame_s in sample_frames:
            frame_s = int(frame_s)
            pe_path = os.path.join(pe_dir, f'bw_{frame_s}{filetype}')
            I = cv2.imread(pe_path, cv2.IMREAD_GRAYSCALE)
            if I is None:
                continue
            I = I[roi[0]:roi[1], roi[2]:roi[3]]
            f_s = F_linked[F_linked.frame == frame_s].copy()
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(src.plot_contacts(I, f_s, F_contact[F_contact.frame == frame_s],
                                        f_s.particle[f_s.boundary].to_numpy()))
            ax.axis('off')
            ax.set_title(f'Contacts — frame {frame_s}')
            out = os.path.join(diag_dir, f'contacts_frame{frame_s}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[contact] diagnostic → {out}')

    return F_contact


# ── main ──────────────────────────────────────────────────────────────────
def _run_one(args, img_dir, src):
    """Run the full pipeline for a single img_dir."""
    args.img_dir = img_dir
    norm = os.path.normpath(img_dir)
    exp_folder = os.path.basename(norm)
    exp_parent = os.path.basename(os.path.dirname(norm))
    # Use parent_subfolder as the pkl stem when the leaf is a generic name like speed1
    exp_stem = f'{exp_parent}_{exp_folder}' if exp_folder.lower().startswith('speed') else exp_folder
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f'Image directory not found: {img_dir}')

    camera_align = make_camera_align()

    diag_dir = os.path.join(args.pkl_dir, f'{exp_stem}_diagnostics')
    if args.verbose:
        os.makedirs(diag_dir, exist_ok=True)

    pkl_path  = os.path.join(args.pkl_dir,  f'{exp_stem}.pkl')
    bond_path = os.path.join(args.bond_dir, f'CONTACT_BOND_{exp_stem}.pkl')

    if not args.skip_tracking:
        print(f'\n=== Step 1: Disk Tracking  [{exp_stem}] ===')
        F_linked = run_tracking(args, camera_align, diag_dir, src)
        F_linked.to_pickle(pkl_path)
        print(f'[tracking] saved → {pkl_path}')
    else:
        print(f'[tracking] skipped — loading {pkl_path}')
        F_linked = pd.read_pickle(pkl_path)

    if not args.skip_contact:
        print(f'\n=== Step 2: Contact Detection  [{exp_stem}] ===')
        F_contact = run_contact(args, camera_align, F_linked, diag_dir, src)
        F_contact.to_pickle(bond_path)
        print(f'[contact] saved → {bond_path}')

    print(f'[done] {exp_stem}\n')


def main():
    args = resolve_args()

    # Normalise to a list so single string and list both work
    dirs = args.img_dir if isinstance(args.img_dir, list) else [args.img_dir]
    dirs = [d for d in dirs if d]  # drop empty strings
    if not dirs:
        raise ValueError(
            'No image directory provided. Set DIRECT_RUN_IMG_DIR for direct run, '
            'or pass --img-dir from CLI.'
        )

    # put pipeline src on path once
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import src

    for img_dir in dirs:
        print(f'\n{"="*60}\nProcessing: {img_dir}\n{"="*60}')
        _run_one(args, img_dir, src)

    print('\nAll done.')


if __name__ == '__main__':
    main()
