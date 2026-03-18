from __future__ import annotations

from typing import Iterable, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spherical_canonicalizer import SphericalTapCanonicalizer


class StencilConv3d(nn.Module):
    """
    Convolution layer using sample-specific 3D interpolation stencils.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stencil_shape: Union[int, Iterable[int]],
        num_taps: int,
        *,
        stride: Union[int, Iterable[int]] = 1,
        padding: Union[int, Iterable[int]] = 0,
        bias: bool = True,
        padding_mode: str = "zeros",
        equator_mask: torch.Tensor | None = None,
        perm_table: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_taps = int(num_taps)

        def _to_3tuple(val: Union[int, Iterable[int]]) -> Tuple[int, int, int]:
            if isinstance(val, int):
                return (int(val), int(val), int(val))
            tpl = tuple(int(v) for v in val)
            if len(tpl) != 3:
                raise ValueError("expected a scalar or iterable of length 3")
            return tpl

        self.stencil_shape = _to_3tuple(stencil_shape)
        self.stride = _to_3tuple(stride)
        self.padding = _to_3tuple(padding)
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                self.num_taps,
                dtype=dtype,
                device=device,
            )
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        if equator_mask is not None and perm_table is not None:
            self.canonicalizer = SphericalTapCanonicalizer(
                equator_mask=equator_mask,
                perm_table=perm_table,
            )
        else:
            self.canonicalizer = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.num_taps
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_taps={self.num_taps}, "
            f"stencil_shape={self.stencil_shape}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"padding_mode='{self.padding_mode}', "
            f"canonicalizer={'yes' if self.canonicalizer is not None else 'no'}"
        )

    def _apply_padding(self, x: torch.Tensor) -> torch.Tensor:
        pd, ph, pw = self.padding
        if pd or ph or pw:
            if self.padding_mode == "zeros":
                return F.pad(
                    x,
                    (pw, pw, ph, ph, pd, pd),
                    mode="constant",
                    value=0.0,
                )
            return F.pad(
                x,
                (pw, pw, ph, ph, pd, pd),
                mode=self.padding_mode,
            )
        return x

    def forward(self, x: torch.Tensor, stencils: torch.Tensor):
        """
        x:        (B, Cin, D, H, W)
        stencils: (B, T, sx, sy, sz)
        """
        if x.ndim != 5:
            raise ValueError("input must have shape (B, Cin, D, H, W)")

        B, Cin, _, _, _ = x.shape
        if Cin != self.in_channels:
            raise ValueError(f"expected input with {self.in_channels} channels, got {Cin}")

        stencils = torch.as_tensor(stencils, dtype=x.dtype, device=x.device)

        sx, sy, sz = self.stencil_shape
        if stencils.shape != (B, self.num_taps, sx, sy, sz):
            raise ValueError(
                f"stencils must have shape ({B}, {self.num_taps}, {sx}, {sy}, {sz})"
            )

        x_padded = self._apply_padding(x)

        BCin = B * Cin
        x_exp = x_padded.reshape(
            1,
            BCin,
            x_padded.shape[2],
            x_padded.shape[3],
            x_padded.shape[4],
        )

        w = stencils[:, None, :, :, :, :].expand(B, Cin, self.num_taps, sx, sy, sz)
        w = w.reshape(BCin * self.num_taps, 1, sx, sy, sz)

        tap_values = F.conv3d(
            x_exp,
            w,
            bias=None,
            stride=self.stride,
            padding=0,
            groups=BCin,
        )

        D_out, H_out, W_out = tap_values.shape[-3:]
        tap_values = tap_values.view(B, Cin, self.num_taps, D_out, H_out, W_out)

        if self.canonicalizer is not None:
            tap_values, final_perm, pole_idx, front_idx = self.canonicalizer(tap_values)
        else:
            final_perm = None
            pole_idx = None
            front_idx = None

        y_out = torch.einsum("bctdhw,oct->bodhw", tap_values, self.weight)

        if self.bias is not None:
            y_out = y_out + self.bias.view(1, -1, 1, 1, 1)

        return y_out, final_perm, pole_idx, front_idx