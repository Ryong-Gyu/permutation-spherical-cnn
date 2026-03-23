from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_package():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "permutation_spherical_cnn",
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load package from repository root")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    pkg = _load_package()

    input_tensor = torch.arange(5 * 5 * 5, dtype=torch.float32).reshape(5, 5, 5)
    input_spacing = 1.0
    output_spacing = 0.5

    output_points, stencil_tensor = pkg.build_cubic_sampling_stencils(
        kernel_size=5,
        grid_size=3,
        grid_spacing=output_spacing,
        cell_spacing=input_spacing,
    )
    output_tensor_template = torch.zeros((3, 3, 3), dtype=torch.float32)

    print("Input tensor shape:", tuple(input_tensor.shape))
    print("Input spacing (angstrom):", input_spacing)
    print("Input cube extent by voxel count (angstrom):", input_tensor.shape[0] * input_spacing)
    print()

    print("Interpolation stencil tensor shape:", tuple(stencil_tensor.shape))
    print("Each stencil kernel shape:", tuple(stencil_tensor.shape[1:]))
    print()

    print("Output tensor shape:", tuple(output_tensor_template.shape))
    print("Output spacing (angstrom):", output_spacing)
    print(
        "Output cube extent by voxel count (angstrom):",
        output_tensor_template.shape[0] * output_spacing,
    )
    print()

    print("Output sampling points shape:", tuple(output_points.shape))
    print("First 5 output sampling points (angstrom):")
    print(output_points[:5])


if __name__ == "__main__":
    main()
