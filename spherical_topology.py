from __future__ import annotations

import itertools
import torch


def normalize_points(points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize 3D points row-wise.

    points: (T, 3)
    """
    points = torch.as_tensor(points, dtype=torch.float32)
    norms = points.norm(dim=1, keepdim=True).clamp_min(eps)
    return points / norms


def build_rotation_matrices_from_axis_permutations() -> torch.Tensor:
    """
    Build the 24 proper rotation matrices of the cube/octahedral group.

    Returns
    -------
    rotations : (24, 3, 3)
    """
    mats = []
    eye = torch.eye(3, dtype=torch.float32)

    for perm in itertools.permutations(range(3)):
        P = eye[list(perm)]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            S = torch.diag(torch.tensor(signs, dtype=torch.float32))
            R = S @ P
            if torch.det(R) > 0.5:
                mats.append(R)

    mats = torch.stack(mats, dim=0)

    # remove duplicates just in case
    uniq = []
    for i in range(mats.shape[0]):
        keep = True
        for j in range(len(uniq)):
            if torch.allclose(mats[i], uniq[j], atol=1e-6, rtol=0):
                keep = False
                break
        if keep:
            uniq.append(mats[i])

    return torch.stack(uniq, dim=0)


def _nearest_index_map(
    rotated_points: torch.Tensor,
    reference_points: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> torch.Tensor:
    """
    Find permutation mapping canonical slot -> source tap index.

    rotated_points:   (T, 3)
    reference_points: (T, 3)

    Returns
    -------
    perm : (T,)
        perm[k] = j means canonical slot k reads from original/source tap j
    """
    T = reference_points.shape[0]

    d2 = ((rotated_points[:, None, :] - reference_points[None, :, :]) ** 2).sum(dim=-1)
    nn_idx = d2.argmin(dim=1)
    nn_dist = d2.min(dim=1).values.sqrt()

    if (nn_dist > atol).any():
        raise ValueError(
            "Rotation does not map points onto the tap set within tolerance. "
            "Your tap set may not be closed under the supplied rotations."
        )

    # uniqueness check
    if torch.unique(nn_idx).numel() != T:
        raise ValueError(
            "Nearest-neighbor mapping is not one-to-one. "
            "Tap set / rotation set pairing is ambiguous."
        )

    return nn_idx


def build_all_permutations_from_rotations(
    sphere_points: torch.Tensor,
    rotations: torch.Tensor,
    *,
    normalize: bool = True,
    atol: float = 1e-5,
    remove_duplicates: bool = True,
) -> torch.Tensor:
    """
    Build permutation bank from spherical tap coordinates and admissible rotations.

    Parameters
    ----------
    sphere_points : (T, 3)
        Coordinates of spherical tap points (center point excluded).
    rotations : (Nrot, 3, 3)
        Admissible rotation matrices.
    normalize : bool
        If True, normalize sphere_points row-wise before matching.
    atol : float
        Tolerance for nearest-point matching.
    remove_duplicates : bool
        If True, duplicate permutations are removed.

    Returns
    -------
    all_permutations : (Nperm, T) long
        Each row is a permutation on spherical taps.
    """
    sphere_points = torch.as_tensor(sphere_points, dtype=torch.float32)
    rotations = torch.as_tensor(rotations, dtype=torch.float32)

    if sphere_points.ndim != 2 or sphere_points.shape[1] != 3:
        raise ValueError("sphere_points must have shape (T, 3)")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (Nrot, 3, 3)")

    ref = normalize_points(sphere_points) if normalize else sphere_points
    perms = []

    for r in range(rotations.shape[0]):
        R = rotations[r]
        rot_pts = (ref @ R.T)
        perm = _nearest_index_map(rot_pts, ref, atol=atol)
        perms.append(perm)

    perms = torch.stack(perms, dim=0)

    if remove_duplicates:
        perms = torch.unique(perms, dim=0)

    return perms