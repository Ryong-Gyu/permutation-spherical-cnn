from __future__ import annotations

from typing import Iterable, Tuple, Union

import math
import torch

from .stencil_generator import LagrangeFilter3D


def _to_3tuple(value: Union[int, float, Iterable[int], Iterable[float]]) -> Tuple[float, float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return (scalar, scalar, scalar)

    triplet = tuple(float(v) for v in value)
    if len(triplet) != 3:
        raise ValueError("expected a scalar or iterable of length 3")
    return triplet


def build_axis_rotation_matrix(
    axis: torch.Tensor,
    angle_degrees: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Build a 3x3 rotation matrix from an axis and an angle in degrees.
    """
    axis = torch.as_tensor(axis, dtype=dtype, device=device)
    if axis.shape != (3,):
        raise ValueError("axis must have shape (3,)")

    axis_norm = axis.norm()
    if axis_norm <= 0:
        raise ValueError("axis must be non-zero")
    axis = axis / axis_norm

    theta = math.radians(float(angle_degrees))
    c = math.cos(theta)
    s = math.sin(theta)
    one_minus_c = 1.0 - c
    x, y, z = axis.tolist()

    return torch.tensor(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=dtype,
        device=device,
    )


def build_cubic_sampling_points(
    grid_size: Union[int, Iterable[int]],
    grid_spacing: Union[float, Iterable[float]],
    *,
    rotation_axis: torch.Tensor | None = None,
    rotation_angle_degrees: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Build a centered cubic sampling grid of shape (nx * ny * nz, 3).

    Parameters
    ----------
    grid_size
        Number of samples on each axis.
    grid_spacing
        Physical spacing on each axis.
    rotation_axis
        Optional axis used for rotational augmentation.
    rotation_angle_degrees
        Rotation angle in degrees applied around ``rotation_axis``.
    """
    nx_f, ny_f, nz_f = _to_3tuple(grid_size)
    nx, ny, nz = int(nx_f), int(ny_f), int(nz_f)
    if (nx, ny, nz) != (nx_f, ny_f, nz_f):
        raise ValueError("grid_size values must be integers")
    if min(nx, ny, nz) < 1:
        raise ValueError("grid_size values must be at least 1")

    sx, sy, sz = _to_3tuple(grid_spacing)
    if min(sx, sy, sz) <= 0:
        raise ValueError("grid_spacing values must be positive")

    x = (torch.arange(nx, dtype=dtype, device=device) - ((nx - 1) / 2.0)) * sx
    y = (torch.arange(ny, dtype=dtype, device=device) - ((ny - 1) / 2.0)) * sy
    z = (torch.arange(nz, dtype=dtype, device=device) - ((nz - 1) / 2.0)) * sz

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    if rotation_axis is not None:
        rotation = build_axis_rotation_matrix(
            rotation_axis,
            rotation_angle_degrees,
            dtype=dtype,
            device=device,
        )
        points = points @ rotation.T

    return points


def build_cubic_sampling_stencils(
    kernel_size: Union[int, Iterable[int]],
    grid_size: Union[int, Iterable[int]],
    grid_spacing: Union[float, Iterable[float]],
    *,
    rotation_axis: torch.Tensor | None = None,
    rotation_angle_degrees: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    """
    Build cubic sampling points and the corresponding Lagrange interpolation stencils.
    """
    spacing = _to_3tuple(grid_spacing)
    cell = torch.diag(torch.tensor(spacing, dtype=dtype, device=device))
    points = build_cubic_sampling_points(
        grid_size=grid_size,
        grid_spacing=spacing,
        rotation_axis=rotation_axis,
        rotation_angle_degrees=rotation_angle_degrees,
        dtype=dtype,
        device=device,
    )
    filters = LagrangeFilter3D(kernel_size, dtype=dtype, device=device)(cell, points)
    return points, filters
