from __future__ import annotations

import torch
import torch.nn as nn


def _check_permutation_rows(perms: torch.Tensor) -> None:
    if perms.ndim != 2:
        raise ValueError("perms must have shape (Nperm, T)")

    nperm, T = perms.shape
    ref = torch.arange(T, device=perms.device)
    sorted_perms, _ = perms.sort(dim=1)

    if not torch.equal(sorted_perms, ref.expand(nperm, T)):
        raise ValueError("Each row of perms must be a valid permutation of [0, ..., T-1]")


def build_equator_and_perm_tables(
    all_permutations: torch.Tensor,
    north_idx: int,
    front_idx: int,
    *,
    fill_invalid_with_identity: bool = True,
):
    """
    Build:
        equator_mask[p, q] = True if q is a valid front candidate when p is pole
        perm_table[p, q] = permutation that sends (pole=p, front=q) to canonical slots
    """
    perms = torch.as_tensor(all_permutations, dtype=torch.long)
    _check_permutation_rows(perms)

    nperm, T = perms.shape

    if not (0 <= north_idx < T):
        raise ValueError(f"north_idx must be in [0, {T-1}]")
    if not (0 <= front_idx < T):
        raise ValueError(f"front_idx must be in [0, {T-1}]")
    if north_idx == front_idx:
        raise ValueError("north_idx and front_idx must be different")

    equator_mask = torch.zeros((T, T), dtype=torch.bool, device=perms.device)
    valid_pair_mask = torch.zeros((T, T), dtype=torch.bool, device=perms.device)

    if fill_invalid_with_identity:
        identity = torch.arange(T, dtype=torch.long, device=perms.device)
        perm_table = identity.view(1, 1, T).expand(T, T, T).clone()
    else:
        perm_table = torch.full((T, T, T), -1, dtype=torch.long, device=perms.device)

    pole_of_perm = perms[:, north_idx]
    front_of_perm = perms[:, front_idx]

    for r in range(nperm):
        p = int(pole_of_perm[r].item())
        f = int(front_of_perm[r].item())
        perm = perms[r]

        equator_mask[p, f] = True

        if valid_pair_mask[p, f]:
            if not torch.equal(perm_table[p, f], perm):
                raise ValueError(
                    f"Ambiguous mapping: multiple different permutations map "
                    f"(pole={p}, front={f}) to canonical ordering."
                )
        else:
            perm_table[p, f] = perm
            valid_pair_mask[p, f] = True

    idx = torch.arange(T, device=perms.device)
    equator_mask[idx, idx] = False
    valid_pair_mask[idx, idx] = False

    return equator_mask, perm_table, valid_pair_mask


def compute_sample_scores(tap_values: torch.Tensor) -> torch.Tensor:
    """
    tap_values: (B, C, T, D, H, W)
    returns:    (B, T)
    """
    if tap_values.ndim != 6:
        raise ValueError("tap_values must have shape (B, C, T, D, H, W)")
    return tap_values.mean(dim=(1, 3, 4, 5))


def choose_perm_from_tables(
    scores: torch.Tensor,
    equator_mask: torch.Tensor,
    perm_table: torch.Tensor,
):
    """
    scores:       (B, T)
    equator_mask: (T, T)
    perm_table:   (T, T, T)
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape (B, T)")

    B, T = scores.shape
    equator_mask = torch.as_tensor(equator_mask, dtype=torch.bool, device=scores.device)
    perm_table = torch.as_tensor(perm_table, dtype=torch.long, device=scores.device)

    if equator_mask.shape != (T, T):
        raise ValueError(f"equator_mask must have shape ({T}, {T})")
    if perm_table.shape != (T, T, T):
        raise ValueError(f"perm_table must have shape ({T}, {T}, {T})")

    pole_idx = scores.argmax(dim=1)

    valid_mask = equator_mask[pole_idx]
    if (~valid_mask).all(dim=1).any():
        raise ValueError("Some samples have no valid equator/front candidate for selected pole.")

    masked_scores = scores.masked_fill(~valid_mask, float("-inf"))
    front_idx = masked_scores.argmax(dim=1)

    final_perm = perm_table[pole_idx, front_idx]
    return final_perm, pole_idx, front_idx


def apply_perm_to_tap_values(
    tap_values: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """
    tap_values: (B, C, T, D, H, W)
    perm:       (B, T)
    """
    if tap_values.ndim != 6:
        raise ValueError("tap_values must have shape (B, C, T, D, H, W)")
    if perm.ndim != 2:
        raise ValueError("perm must have shape (B, T)")

    B, C, T, D, H, W = tap_values.shape
    if perm.shape != (B, T):
        raise ValueError(f"perm must have shape ({B}, {T})")

    gather_idx = perm[:, None, :, None, None, None].expand(B, C, T, D, H, W)
    return tap_values.gather(dim=2, index=gather_idx)


class SphericalTapCanonicalizer(nn.Module):
    def __init__(self, equator_mask: torch.Tensor, perm_table: torch.Tensor):
        super().__init__()

        equator_mask = torch.as_tensor(equator_mask, dtype=torch.bool)
        perm_table = torch.as_tensor(perm_table, dtype=torch.long)

        if equator_mask.ndim != 2:
            raise ValueError("equator_mask must have shape (T, T)")
        if perm_table.ndim != 3:
            raise ValueError("perm_table must have shape (T, T, T)")

        T0, T1 = equator_mask.shape
        P0, P1, P2 = perm_table.shape

        if T0 != T1:
            raise ValueError("equator_mask must be square")
        if (P0, P1, P2) != (T0, T0, T0):
            raise ValueError("perm_table must have shape (T, T, T)")

        self.num_taps = T0
        self.register_buffer("equator_mask", equator_mask)
        self.register_buffer("perm_table", perm_table)

    def choose_perm(self, scores: torch.Tensor):
        return choose_perm_from_tables(scores, self.equator_mask, self.perm_table)

    def forward(self, tap_values: torch.Tensor):
        scores = compute_sample_scores(tap_values)
        final_perm, pole_idx, front_idx = self.choose_perm(scores)
        tap_values_perm = apply_perm_to_tap_values(tap_values, final_perm)
        return tap_values_perm, final_perm, pole_idx, front_idx