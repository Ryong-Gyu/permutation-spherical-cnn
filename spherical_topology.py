from __future__ import annotations

import itertools
import math

import torch


def normalize_points(points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize 3D points row-wise.

    points: (T, 3)
    """
    points = torch.as_tensor(points, dtype=torch.float32)
    norms = points.norm(dim=1, keepdim=True).clamp_min(eps)
    return points / norms


def build_rotation_matrices_about_z(n: int) -> torch.Tensor:
    """
    Build the n proper rotation matrices corresponding to n-fold symmetry
    around the z axis.

    Returns
    -------
    rotations : (n, 3, 3)
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    mats = []
    for k in range(n):
        theta = (2.0 * math.pi * k) / n
        c = math.cos(theta)
        s = math.sin(theta)
        mats.append(
            torch.tensor(
                [
                    [c, -s, 0.0],
                    [s, c, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

    return torch.stack(mats, dim=0)


def build_nfold_symmetric_spherical_grid(
    n: int,
    *,
    radius: float = 1.0,
    azimuth_offset_degrees: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    """
    Build a spherical sampling grid by sweeping spherical coordinates on
    theta/phi steps of pi/n and 2*pi/n respectively.

    The layout is:
        1 center
        1 north pole
        n taps for each intermediate theta = i*pi/n, i=1, ..., n-1
        1 south pole

    For each intermediate theta ring, the azimuth sampling is
        phi_k = offset + 2*pi*k/n,  k = 0, ..., n-1

    Returns
    -------
    points : (n * (n - 1) + 3, 3)
        Cartesian coordinates ordered as [center, north pole, theta-rings..., south pole].
    index_map : dict[str, int | torch.Tensor]
        Convenience indices for the grid layout. ``theta_rings[i - 1]`` corresponds
        to the ring sampled at ``theta = i*pi/n``.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if radius <= 0:
        raise ValueError("radius must be positive")

    dtype = torch.float32 if dtype is None else dtype

    az0 = math.radians(azimuth_offset_degrees)
    angles = az0 + (2.0 * math.pi / n) * torch.arange(n, dtype=dtype, device=device)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    def _ring(theta: float) -> torch.Tensor:
        xy_radius = radius * math.sin(theta)
        z_value = radius * math.cos(theta)
        z = torch.full((n,), z_value, dtype=dtype, device=device)
        x = xy_radius * cos_a
        y = xy_radius * sin_a
        return torch.stack([x, y, z], dim=1)

    center = torch.zeros((1, 3), dtype=dtype, device=device)
    north_pole = torch.tensor([[0.0, 0.0, radius]], dtype=dtype, device=device)
    rings = []
    ring_indices = []

    start = 2
    for i in range(1, n):
        theta = (math.pi * i) / n
        ring = _ring(theta)
        rings.append(ring)
        ring_indices.append(torch.arange(start, start + n, device=device))
        start += n

    south_pole = torch.tensor([[0.0, 0.0, -radius]], dtype=dtype, device=device)

    points = torch.cat(
        [center, north_pole, *rings, south_pole],
        dim=0,
    )

    theta_values = torch.linspace(0.0, math.pi, steps=n + 1, dtype=dtype, device=device)
    phi_values = angles.remainder(2.0 * math.pi)
    index_map = {
        "center": 0,
        "north_pole": 1,
        "south_pole": start,
        "rings": tuple(ring_indices),
        "theta_rings": tuple(ring_indices),
        "surface": torch.arange(1, start + 1, device=device),
        "theta_values": theta_values,
        "phi_values": phi_values,
    }

    return points, index_map


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
