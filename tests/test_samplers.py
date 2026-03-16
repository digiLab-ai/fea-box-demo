from __future__ import annotations

from digilab_samplers import sample_parameter_space


def test_lhs_sampler_returns_expected_shape_and_columns():
    df = sample_parameter_space(method="lhs", n_samples=10, seed=7)
    assert df.shape == (10, 5)
    assert list(df.columns) == [
        "thermal_conductivity",
        "heat_source_power",
        "heat_source_distance",
        "convective_h",
        "initial_temperature",
    ]


def test_mc_sampler_returns_expected_shape_and_columns():
    df = sample_parameter_space(method="mc", n_samples=10, seed=7)
    assert df.shape == (10, 5)
    assert list(df.columns) == [
        "thermal_conductivity",
        "heat_source_power",
        "heat_source_distance",
        "convective_h",
        "initial_temperature",
    ]


def test_sobol_sampler_returns_expected_shape_and_columns():
    df = sample_parameter_space(method="sobol", n_samples=10, seed=7)
    assert df.shape == (10, 5)
    assert list(df.columns) == [
        "thermal_conductivity",
        "heat_source_power",
        "heat_source_distance",
        "convective_h",
        "initial_temperature",
    ]
