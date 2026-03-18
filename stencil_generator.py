from __future__ import annotations

from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn


class LagrangeFilter3D(nn.Module):
    """
    Generate 3D tensor-product Lagrange interpolation filters.
    """

    def __init__(
        self,
        kernel_size: Union[int, Iterable[int]],
        *,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size: Tuple[int, int, int] = (int(kernel_size),) * 3
        else:
            ks = tuple(int(k) for k in kernel_size)
            if len(ks) != 3:
                raise ValueError("kernel_size must be an int or an iterable of length 3")
            self.kernel_size = ks

        for k in self.kernel_size:
            if k < 2:
                raise ValueError("each dimension of kernel_size must be at least 2")

        self.dtype = dtype
        self.device = device

    def _lagrange_basis_1d(self, x: torch.Tensor, n: int) -> torch.Tensor:
        nodes = torch.arange(n, dtype=x.dtype, device=x.device)
        basis = []

        for i in range(n):
            w = torch.ones_like(x)
            xi = nodes[i]
            for j in range(n):
                if j == i:
                    continue
                xj = nodes[j]
                w = w * (x - xj) / (xi - xj)
            basis.append(w)

        return torch.stack(basis, dim=-1)

    def forward(self, cell: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        cell = torch.as_tensor(cell, dtype=self.dtype, device=self.device)
        target_points = torch.as_tensor(target_points, dtype=self.dtype, device=self.device)

        if cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        if target_points.ndim != 2 or target_points.shape[1] != 3:
            raise ValueError("target_points must have shape (Nt, 3)")

        kx, ky, kz = self.kernel_size

        inv_cell = torch.inverse(cell)
        frac = target_points @ inv_cell.T

        anchors = torch.floor(frac).to(torch.long) - torch.tensor(
            [kx // 2, ky // 2, kz // 2],
            dtype=torch.long,
            device=frac.device,
        )
        local = frac - anchors.to(self.dtype)

        wx = self._lagrange_basis_1d(local[:, 0], kx)
        wy = self._lagrange_basis_1d(local[:, 1], ky)
        wz = self._lagrange_basis_1d(local[:, 2], kz)

        filters = (
            wx[:, :, None, None]
            * wy[:, None, :, None]
            * wz[:, None, None, :]
        )

        return filters