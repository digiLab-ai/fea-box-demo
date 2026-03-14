from __future__ import annotations

import pandas as pd

from digilab_simulators.simulators import CubeThermalSteadyStateConfig, simulator_factory


def test_cube_thermal_simulator_forward_runs():
    config = CubeThermalSteadyStateConfig(
        nx=6,
        ny=5,
        nz=4,
        max_iterations=500,
        tolerance=1e-5,
    )
    simulator = simulator_factory(config)

    X = pd.DataFrame(
        [
            {
                "thermal_conductivity": 15.0,
                "volumetric_heat_source": 1.2e4,
                "heat_flux_x0": 8.0e3,
                "convective_h": 10.0,
                "initial_temperature": 293.15,
            }
        ]
    )

    outputs = simulator.forward(X)
    assert len(outputs) == 1

    out = outputs[0]
    assert out["nx"] == 6
    assert out["ny"] == 5
    assert out["nz"] == 4
    assert len(out["points"]) == 6 * 5 * 4
    assert len(out["point_data"]["temperature"]) == 6 * 5 * 4
    assert out["max_temperature"] >= out["min_temperature"]
    assert out["temperature_range"] >= 0.0


def test_summary_dataframe_contains_expected_columns():
    config = CubeThermalSteadyStateConfig()
    simulator = simulator_factory(config)
    X = pd.DataFrame(
        [
            {
                "thermal_conductivity": 12.0,
                "volumetric_heat_source": 2.0e4,
                "heat_flux_x0": 1.5e4,
                "convective_h": 8.0,
                "initial_temperature": 293.15,
            }
        ]
    )
    outputs = simulator.forward(X)
    df = simulator.outputs_to_summary_dataframe(outputs)
    assert "centre_temperature" in df.columns
    assert "temperature_range" in df.columns
    assert len(df) == 1
