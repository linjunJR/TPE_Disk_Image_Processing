
import numpy as np
import pandas as pd
from scipy.spatial import distance

def process_singular_bonds(F_out, boundary_pid):
    F_out = F_out.copy()
    boundary_pid = np.array(boundary_pid)
    F_out['singular'] = False

    all_particles = pd.concat([F_out['i'], F_out['j']])
    counts = all_particles.value_counts()

    bulk_particles = counts.index.difference(boundary_pid)
    singular_particles = bulk_particles[counts[bulk_particles] == 1]
    singular_set = set(singular_particles)

    i_singular = F_out['i'].isin(singular_set)
    j_singular = F_out['j'].isin(singular_set)
    F_out = F_out[~(i_singular & j_singular)].reset_index(drop=True)

    # Re-compute masks after row removal
    i_sing = F_out['i'].isin(singular_set)
    j_sing = F_out['j'].isin(singular_set)
    F_out['singular'] = np.where(i_sing, F_out['i'],
                        np.where(j_sing, F_out['j'], -1)).astype(int)
    return F_out


def drop_bulk_duplicate(df, boundary_pid):
    bulk_mask = (~df['i'].isin(boundary_pid)) & (~df['j'].isin(boundary_pid))
    df['pair_key'] = list(zip(df[['i', 'j']].min(axis=1), df[['i', 'j']].max(axis=1)))
    keep_mask = ~(bulk_mask & df.duplicated('pair_key'))
    df = df[keep_mask].reset_index(drop=True)
    return df.drop(columns='pair_key')




def duplicate_and_swap_bulk(F_out):
    """Clone bulk rows and swap i↔j so every particle has its contacts listed under i."""
    to_duplicate = F_out[~F_out['j_on_boundary']].copy()
    swap_cols = {
        'i': 'j', 'xi': 'xj', 'yi': 'yj', 'ri': 'rj',
        'j': 'i', 'xj': 'xi', 'yj': 'yi', 'rj': 'ri'
    }
    swap_cols = {k: v for k, v in swap_cols.items() if k in F_out.columns}
    other_cols = [c for c in F_out.columns if c not in swap_cols]
    swapped = to_duplicate.rename(columns=swap_cols)[[*swap_cols.keys(), *other_cols]]
    return pd.concat([F_out, swapped], ignore_index=True)


def get_all_bonds(f, boundary_pid, d_tol):
    """Return all candidate contacts within r_i + r_j + d_tol."""
    coords = f[['x', 'y']].to_numpy()
    radii  = f['rpx'].to_numpy()
    pid    = f['particle'].to_numpy()

    dist_mat = distance.cdist(coords, coords)
    r_mat    = radii[:, None] + radii[None, :]
    nbrs_bool = (dist_mat < r_mat + d_tol) & (dist_mat > 0)
    nbr_i, nbr_j = np.where(nbrs_bool)

    result = np.empty((len(nbr_i), 8), dtype=float)
    result[:, 0]   = pid[nbr_i]
    result[:, 1:3] = coords[nbr_i]
    result[:, 3]   = radii[nbr_i]
    result[:, 4]   = pid[nbr_j]
    result[:, 5:7] = coords[nbr_j]
    result[:, 7]   = radii[nbr_j]

    F_out = pd.DataFrame(result, columns=['i', 'xi', 'yi', 'ri', 'j', 'xj', 'yj', 'rj'])
    F_out[['i', 'j']] = F_out[['i', 'j']].astype(int)
    F_out = F_out[~F_out['i'].isin(set(boundary_pid))].reset_index(drop=True)
    F_out = drop_bulk_duplicate(F_out, boundary_pid)
    F_out['j_on_boundary'] = F_out['j'].isin(boundary_pid)
    F_out = F_out.sort_values(by='i').reset_index(drop=True)
    return F_out, f, nbr_i, nbr_j
