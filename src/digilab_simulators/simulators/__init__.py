from .base import Simulator, SimulatorConfig, SimulatorMeta
from .fea_box import (
    CubeThermalSteadyStateConfig,
    CubeThermalSteadyStateOutput,
    CubeThermalSteadyStateSimulator,
)


def simulator_factory(config: SimulatorConfig) -> Simulator:
    if isinstance(config, CubeThermalSteadyStateConfig):
        return CubeThermalSteadyStateSimulator(config)
    raise TypeError(f"Unsupported simulator config type: {type(config)!r}")


__all__ = [
    "Simulator",
    "SimulatorConfig",
    "SimulatorMeta",
    "CubeThermalSteadyStateConfig",
    "CubeThermalSteadyStateOutput",
    "CubeThermalSteadyStateSimulator",
    "simulator_factory",
]
