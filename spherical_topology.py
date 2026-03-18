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


def build_minimal_rotation_matrix(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build a proper rotation matrix that maps `source` onto `target`.

    For the antiparallel case, a deterministic orthogonal fallback axis is used.
    """
    source = normalize_points(torch.as_tensor(source, dtype=torch.float32).view(1, 3))[0]
    target = normalize_points(torch.as_tensor(target, dtype=torch.float32).view(1, 3))[0]

    dot = torch.clamp((source * target).sum(), -1.0, 1.0)
    eye = torch.eye(3, dtype=torch.float32)

    if dot > 1.0 - eps:
        return eye

    if dot < -1.0 + eps:
        basis = torch.eye(3, dtype=torch.float32)
        axis = basis[int(source.abs().argmin().item())]
        axis = axis - (axis * source).sum() * source
        axis = normalize_points(axis.view(1, 3))[0]
        return (2.0 * torch.outer(axis, axis)) - eye

    v = torch.cross(source, target, dim=0)
    s = v.norm()
    vx = torch.tensor(
        [
            [0.0, -v[2].item(), v[1].item()],
            [v[2].item(), 0.0, -v[0].item()],
            [-v[1].item(), v[0].item(), 0.0],
        ],
        dtype=torch.float32,
    )
    return eye + vx + (vx @ vx) * ((1.0 - dot) / (s * s))


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


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """
    Invert a permutation with convention output[k] = input[perm[k]].
    """
    perm = torch.as_tensor(perm, dtype=torch.long)
    if perm.ndim == 1:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return inv
    if perm.ndim == 2:
        inv = torch.empty_like(perm)
        src = torch.arange(perm.shape[1], device=perm.device).expand_as(perm)
        inv.scatter_(1, perm, src)
        return inv
    raise ValueError("perm must have shape (T,) or (N, T)")


def compose_permutations(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """
    Compose permutations under the convention used by apply_perm_to_tap_values.

    If `first` is applied and then `second` is applied, the composed permutation is
    `first[second]`.
    """
    first = torch.as_tensor(first, dtype=torch.long)
    second = torch.as_tensor(second, dtype=torch.long, device=first.device)

    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("first and second must both have shape (T,)")
    if first.shape != second.shape:
        raise ValueError("first and second must have the same shape")

    return first[second]


def build_tilt_permutations_to_north(
    sphere_points: torch.Tensor,
    *,
    north_idx: int,
    candidate_pole_indices: torch.Tensor | None = None,
    atol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one tilt permutation per candidate pole.

    Each tilt is the nearest-neighbor permutation induced by the minimal proper
    rotation that aligns the candidate pole to the canonical north tap. This
    requires the supplied tap set to be closed under those tilts within `atol`.
    """
    sphere_points = torch.as_tensor(sphere_points, dtype=torch.float32)
    if sphere_points.ndim != 2 or sphere_points.shape[1] != 3:
        raise ValueError("sphere_points must have shape (T, 3)")

    T = sphere_points.shape[0]
    if not (0 <= north_idx < T):
        raise ValueError(f"north_idx must be in [0, {T-1}]")

    if candidate_pole_indices is None:
        candidate_pole_indices = torch.arange(T, device=sphere_points.device)
    else:
        candidate_pole_indices = torch.as_tensor(
            candidate_pole_indices,
            dtype=torch.long,
            device=sphere_points.device,
        )

    ref = normalize_points(sphere_points)
    north_point = ref[north_idx]

    rotations = []
    perms = []
    for pole_idx in candidate_pole_indices.tolist():
        R = build_minimal_rotation_matrix(ref[pole_idx], north_point)
        perm = _nearest_index_map(ref @ R.T, ref, atol=atol)
        rotations.append(R)
        perms.append(perm)

    return (
        candidate_pole_indices,
        torch.stack(rotations, dim=0),
        torch.stack(perms, dim=0),
    )


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


def build_tilt_spin_permutation_bank(
    sphere_points: torch.Tensor,
    *,
    north_idx: int,
    spin_rotations: torch.Tensor,
    candidate_pole_indices: torch.Tensor | None = None,
    atol: float = 1e-5,
    remove_duplicates: bool = True,
):
    """
    Build a factorized permutation bank with the structure:

        tilt permutation sending a selected pole to north
        followed by
        z-spin permutation selecting the front direction.

    Returns
    -------
    all_permutations : (N, T)
        Combined permutation bank.
    metadata : dict[str, torch.Tensor]
        Metadata containing candidate poles, tilt permutations, spin permutations,
        tilt rotations, and the pole/spin index associated with each combined row.

    Notes
    -----
    This utility assumes the supplied tap set is closed under both the tilt
    rotations and the provided spin rotations, otherwise permutation construction
    will fail with a ValueError.
    """
    sphere_points = torch.as_tensor(sphere_points, dtype=torch.float32)
    spin_rotations = torch.as_tensor(spin_rotations, dtype=torch.float32)

    candidate_poles, tilt_rotations, tilt_perms = build_tilt_permutations_to_north(
        sphere_points,
        north_idx=north_idx,
        candidate_pole_indices=candidate_pole_indices,
        atol=atol,
    )
    spin_perms = build_all_permutations_from_rotations(
        sphere_points,
        spin_rotations,
        normalize=True,
        atol=atol,
        remove_duplicates=remove_duplicates,
    )

    combined_perms = []
    combined_poles = []
    combined_spins = []

    for pole_row, pole_idx in enumerate(candidate_poles.tolist()):
        tilt_perm = tilt_perms[pole_row]
        for spin_row in range(spin_perms.shape[0]):
            perm = compose_permutations(tilt_perm, spin_perms[spin_row])
            combined_perms.append(perm)
            combined_poles.append(pole_idx)
            combined_spins.append(spin_row)

    all_permutations = torch.stack(combined_perms, dim=0)

    if remove_duplicates:
        unique_perms, inverse = torch.unique(
            all_permutations,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        first_occurrence = torch.full(
            (unique_perms.shape[0],),
            fill_value=-1,
            dtype=torch.long,
            device=all_permutations.device,
        )
        for row, uniq_idx in enumerate(inverse.tolist()):
            if first_occurrence[uniq_idx] < 0:
                first_occurrence[uniq_idx] = row

        all_permutations = unique_perms
        combined_poles = torch.tensor(combined_poles, dtype=torch.long, device=all_permutations.device)[
            first_occurrence
        ]
        combined_spins = torch.tensor(combined_spins, dtype=torch.long, device=all_permutations.device)[
            first_occurrence
        ]
    else:
        combined_poles = torch.tensor(combined_poles, dtype=torch.long, device=all_permutations.device)
        combined_spins = torch.tensor(combined_spins, dtype=torch.long, device=all_permutations.device)

    metadata = {
        "candidate_pole_indices": candidate_poles,
        "tilt_rotations": tilt_rotations,
        "tilt_permutations": tilt_perms,
        "spin_permutations": spin_perms,
        "combined_pole_indices": combined_poles,
        "combined_spin_indices": combined_spins,
    }
    return all_permutations, metadata
