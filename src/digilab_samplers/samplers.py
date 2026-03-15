from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float


DEFAULT_PARAMETER_SPECS = [
    ParameterSpec("thermal_conductivity", 5.0, 25.0),
    ParameterSpec("volumetric_heat_source", 5.0e3, 4.0e4),
    ParameterSpec("heat_flux_x0", 5.0e3, 2.5e4),
    ParameterSpec("convective_h", 2.0, 20.0),
    ParameterSpec("initial_temperature", 285.15, 315.15),
]


def sample_parameter_space(
    *,
    method: str,
    n_samples: int,
    seed: int = 42,
    specs: list[ParameterSpec] | None = None,
) -> pd.DataFrame:
    parameter_specs = DEFAULT_PARAMETER_SPECS if specs is None else specs
    dimension = len(parameter_specs)
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    if method == "lhs":
        unit_samples = _latin_hypercube(n_samples=n_samples, dimension=dimension, seed=seed)
    elif method == "sobol":
        unit_samples = _sobol_sequence(n_samples=n_samples, dimension=dimension)
    else:
        raise ValueError(f"Unsupported sampling method: {method}")

    scaled = {}
    for idx, spec in enumerate(parameter_specs):
        scaled[spec.name] = spec.lower + unit_samples[:, idx] * (spec.upper - spec.lower)
    return pd.DataFrame(scaled)


def _latin_hypercube(*, n_samples: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, dimension), dtype=float)
    for dim in range(dimension):
        perm = rng.permutation(n_samples)
        samples[:, dim] = (perm + rng.random(n_samples)) / n_samples
    return samples


def _sobol_sequence(*, n_samples: int, dimension: int) -> np.ndarray:
    if dimension > 5:
        raise ValueError("This demo Sobol sampler supports up to 5 dimensions.")

    direction_params = [
        None,
        (1, 0, [1]),
        (2, 1, [1, 3]),
        (3, 1, [1, 3, 1]),
        (3, 2, [1, 1, 1]),
    ]
    max_bits = max(1, int(np.ceil(np.log2(max(2, n_samples)))))
    directions = np.zeros((dimension, max_bits), dtype=np.uint32)

    for bit in range(max_bits):
        directions[0, bit] = 1 << (31 - bit)

    for dim in range(1, dimension):
        s, a, m = direction_params[dim]
        for bit in range(max_bits):
            if bit < s:
                directions[dim, bit] = m[bit] << (31 - bit)
            else:
                value = directions[dim, bit - s] ^ (directions[dim, bit - s] >> s)
                for k in range(1, s):
                    if (a >> (s - 1 - k)) & 1:
                        value ^= directions[dim, bit - k]
                directions[dim, bit] = value

    sequence = np.zeros((n_samples, dimension), dtype=float)
    state = np.zeros(dimension, dtype=np.uint32)
    scale = float(2**32)
    for idx in range(n_samples):
        if idx > 0:
            zero_bit = _rightmost_zero_bit(idx - 1)
            state ^= directions[:, zero_bit]
        sequence[idx] = state / scale
    return sequence


def _rightmost_zero_bit(value: int) -> int:
    bit = 0
    while value & 1:
        value >>= 1
        bit += 1
    return bit
