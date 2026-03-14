from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from pydantic import Field
from typeguard import typechecked

from .base import Simulator, SimulatorConfig, SimulatorMeta


DEFAULT_NX = 8
DEFAULT_NY = 8
DEFAULT_NZ = 8
DEFAULT_LENGTH_X = 1.0
DEFAULT_LENGTH_Y = 1.0
DEFAULT_LENGTH_Z = 1.0
DEFAULT_AMBIENT_TEMPERATURE = 293.15
DEFAULT_MAX_ITERATIONS = 4000
DEFAULT_TOLERANCE = 1e-6
DEFAULT_INITIALISATION = "ambient"

VTK_TETRA = 10


class CubeThermalSteadyStateConfig(SimulatorConfig):
    """Configuration for CubeThermalSteadyStateSimulator."""

    nx: int = Field(default=DEFAULT_NX, ge=2, description="Number of nodes in x direction.")
    ny: int = Field(default=DEFAULT_NY, ge=2, description="Number of nodes in y direction.")
    nz: int = Field(default=DEFAULT_NZ, ge=2, description="Number of nodes in z direction.")

    length_x: float = Field(default=DEFAULT_LENGTH_X, gt=0, description="Cube length in x direction.")
    length_y: float = Field(default=DEFAULT_LENGTH_Y, gt=0, description="Cube length in y direction.")
    length_z: float = Field(default=DEFAULT_LENGTH_Z, gt=0, description="Cube length in z direction.")

    ambient_temperature: float = Field(default=DEFAULT_AMBIENT_TEMPERATURE, description="Ambient/reference temperature.")
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1, description="Maximum number of steady-state iterations.")
    tolerance: float = Field(default=DEFAULT_TOLERANCE, gt=0, description="Convergence tolerance on maximum nodal change.")

    initialisation: Literal["ambient", "linear_hot_to_cold"] = Field(
        default=DEFAULT_INITIALISATION,
        description="Initial field guess used before iteration.",
    )

    meta: SimulatorMeta = Field(
        default=SimulatorMeta(
            name="CubeThermalSteadyStateSimulator",
            description="Mock steady-state cube thermal solver with VTU-compatible tetrahedral mesh output",
            version="0.1.0",
            tags=["thermal", "heat-transfer", "3d", "mesh", "tetra", "vtu", "uq"],
        )
    )


class CubeThermalSteadyStateOutput(TypedDict):
    sample_index: int
    converged: bool
    iterations: int
    max_delta: float
    nx: int
    ny: int
    nz: int
    length_x: float
    length_y: float
    length_z: float
    ambient_temperature: float
    max_temperature: float
    min_temperature: float
    mean_temperature: float
    centre_temperature: float
    heated_face_mean_temperature: float
    opposite_face_mean_temperature: float
    temperature_range: float
    input_parameters: dict[str, float]
    points: list[list[float]]
    cells: list[list[int]]
    cell_types: list[int]
    point_data: dict[str, list[float]]
    cell_data: dict[str, list[float]]


@typechecked
class CubeThermalSteadyStateSimulator(Simulator):
    """Mock steady-state thermal cube simulator."""

    REQUIRED_INPUT_COLUMNS = [
        "thermal_conductivity",
        "volumetric_heat_source",
        "heat_flux_x0",
        "convective_h",
        "initial_temperature",
    ]

    OPTIONAL_INPUT_COLUMNS = [
        "ambient_temperature",
        "length_x",
        "length_y",
        "length_z",
    ]

    def __init__(self, config: CubeThermalSteadyStateConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.nz = config.nz
        self.length_x = config.length_x
        self.length_y = config.length_y
        self.length_z = config.length_z
        self.dx = self.length_x / (self.nx - 1)
        self.dy = self.length_y / (self.ny - 1)
        self.dz = self.length_z / (self.nz - 1)
        self.ambient_temperature = config.ambient_temperature
        self.max_iterations = config.max_iterations
        self.tolerance = config.tolerance
        self.initialisation = config.initialisation
        self._points = self._build_points()
        self._cells = self._build_tet_cells()
        self._cell_types = [VTK_TETRA] * len(self._cells)

    def forward(self, X: pd.DataFrame | list[dict[str, float]] | list[list[float]]) -> list[CubeThermalSteadyStateOutput]:
        X_df = self._coerce_inputs_to_dataframe(X)
        return self.evaluate_dataframe(X_df)

    def evaluate_dataframe(self, X: pd.DataFrame) -> list[CubeThermalSteadyStateOutput]:
        self._validate_input_dataframe(X)
        outputs: list[CubeThermalSteadyStateOutput] = []
        for sample_index, (_, row) in enumerate(X.iterrows()):
            outputs.append(
                self.evaluate(
                    sample_index=sample_index,
                    thermal_conductivity=float(row["thermal_conductivity"]),
                    volumetric_heat_source=float(row["volumetric_heat_source"]),
                    heat_flux_x0=float(row["heat_flux_x0"]),
                    convective_h=float(row["convective_h"]),
                    initial_temperature=float(row["initial_temperature"]),
                    ambient_temperature=float(row.get("ambient_temperature", self.ambient_temperature)),
                    length_x=float(row.get("length_x", self.length_x)),
                    length_y=float(row.get("length_y", self.length_y)),
                    length_z=float(row.get("length_z", self.length_z)),
                )
            )
        return outputs

    def evaluate(
        self,
        *,
        sample_index: int,
        thermal_conductivity: float,
        volumetric_heat_source: float,
        heat_flux_x0: float,
        convective_h: float,
        initial_temperature: float,
        ambient_temperature: float | None = None,
        length_x: float | None = None,
        length_y: float | None = None,
        length_z: float | None = None,
    ) -> CubeThermalSteadyStateOutput:
        if thermal_conductivity <= 0:
            raise ValueError("thermal_conductivity must be positive.")
        if convective_h < 0:
            raise ValueError("convective_h must be non-negative.")
        if ambient_temperature is None:
            ambient_temperature = self.ambient_temperature

        lx = self.length_x if length_x is None else length_x
        ly = self.length_y if length_y is None else length_y
        lz = self.length_z if length_z is None else length_z
        if lx <= 0 or ly <= 0 or lz <= 0:
            raise ValueError("All lengths must be positive.")

        dx = lx / (self.nx - 1)
        dy = ly / (self.ny - 1)
        dz = lz / (self.nz - 1)

        points = self._build_points(length_x=lx, length_y=ly, length_z=lz)
        cells = self._build_tet_cells()
        cell_types = [VTK_TETRA] * len(cells)

        T = self._initial_temperature_field(
            initial_temperature=initial_temperature,
            ambient_temperature=ambient_temperature,
        )

        converged = False
        max_delta = np.inf
        iterations = 0
        for iteration in range(1, self.max_iterations + 1):
            T_new = T.copy()
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        T_new[i, j, k] = self._update_node_temperature(
                            T=T,
                            i=i,
                            j=j,
                            k=k,
                            thermal_conductivity=thermal_conductivity,
                            volumetric_heat_source=volumetric_heat_source,
                            heat_flux_x0=heat_flux_x0,
                            convective_h=convective_h,
                            ambient_temperature=ambient_temperature,
                            dx=dx,
                            dy=dy,
                            dz=dz,
                        )
            max_delta = float(np.max(np.abs(T_new - T)))
            T = T_new
            iterations = iteration
            if max_delta < self.tolerance:
                converged = True
                break

        flat_temperature = self._flatten_temperature_field(T)
        centre_temperature = float(T[self.nx // 2, self.ny // 2, self.nz // 2])
        heated_face_mean_temperature = float(np.mean(T[0, :, :]))
        opposite_face_mean_temperature = float(np.mean(T[-1, :, :]))

        point_data = {"temperature": flat_temperature.tolist()}
        cell_temperature = self._compute_cell_temperature_from_points(cells=cells, point_temperature=flat_temperature)
        cell_data = {"temperature": cell_temperature.tolist()}

        return {
            "sample_index": sample_index,
            "converged": converged,
            "iterations": iterations,
            "max_delta": max_delta,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "length_x": float(lx),
            "length_y": float(ly),
            "length_z": float(lz),
            "ambient_temperature": float(ambient_temperature),
            "max_temperature": float(np.max(flat_temperature)),
            "min_temperature": float(np.min(flat_temperature)),
            "mean_temperature": float(np.mean(flat_temperature)),
            "centre_temperature": centre_temperature,
            "heated_face_mean_temperature": heated_face_mean_temperature,
            "opposite_face_mean_temperature": opposite_face_mean_temperature,
            "temperature_range": float(np.max(flat_temperature) - np.min(flat_temperature)),
            "input_parameters": {
                "thermal_conductivity": float(thermal_conductivity),
                "volumetric_heat_source": float(volumetric_heat_source),
                "heat_flux_x0": float(heat_flux_x0),
                "convective_h": float(convective_h),
                "initial_temperature": float(initial_temperature),
                "ambient_temperature": float(ambient_temperature),
                "length_x": float(lx),
                "length_y": float(ly),
                "length_z": float(lz),
            },
            "points": points.tolist(),
            "cells": cells,
            "cell_types": cell_types,
            "point_data": point_data,
            "cell_data": cell_data,
        }

    def outputs_to_summary_dataframe(self, outputs: list[CubeThermalSteadyStateOutput]) -> pd.DataFrame:
        records = []
        for out in outputs:
            record = {
                "sample_index": out["sample_index"],
                "converged": out["converged"],
                "iterations": out["iterations"],
                "max_delta": out["max_delta"],
                "max_temperature": out["max_temperature"],
                "min_temperature": out["min_temperature"],
                "mean_temperature": out["mean_temperature"],
                "centre_temperature": out["centre_temperature"],
                "heated_face_mean_temperature": out["heated_face_mean_temperature"],
                "opposite_face_mean_temperature": out["opposite_face_mean_temperature"],
                "temperature_range": out["temperature_range"],
            }
            record.update(out["input_parameters"])
            records.append(record)
        return pd.DataFrame.from_records(records)

    def output_to_point_dataframe(self, output: CubeThermalSteadyStateOutput) -> pd.DataFrame:
        points = np.asarray(output["points"], dtype=float)
        temperature = np.asarray(output["point_data"]["temperature"], dtype=float)
        return pd.DataFrame(
            {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2], "temperature": temperature}
        )

    def mesh_nodes_to_dataframe(
        self,
        *,
        length_x: float | None = None,
        length_y: float | None = None,
        length_z: float | None = None,
    ) -> pd.DataFrame:
        points = self._build_points(length_x=length_x, length_y=length_y, length_z=length_z)
        return pd.DataFrame(points, columns=["x", "y", "z"])

    def mesh_elements_to_dataframe(self) -> pd.DataFrame:
        cells = self._build_tet_cells()
        column_names = [f"node_{idx}" for idx in range(4)]
        return pd.DataFrame(cells, columns=column_names, dtype=int)

    def _coerce_inputs_to_dataframe(self, X: pd.DataFrame | list[dict[str, float]] | list[list[float]]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, list) and len(X) == 0:
            raise ValueError("X must contain at least one sample.")
        if isinstance(X, list) and isinstance(X[0], dict):
            return pd.DataFrame(X)
        if isinstance(X, list) and isinstance(X[0], list):
            if len(X[0]) != len(self.REQUIRED_INPUT_COLUMNS):
                raise ValueError(
                    "When passing list[list[float]], each row must match the required column order: "
                    f"{self.REQUIRED_INPUT_COLUMNS}"
                )
            return pd.DataFrame(X, columns=self.REQUIRED_INPUT_COLUMNS)
        raise TypeError("X must be a pandas DataFrame, list[dict[str, float]], or list[list[float]].")

    def _validate_input_dataframe(self, X: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_INPUT_COLUMNS if col not in X.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing required columns: {missing}")
        if len(X) == 0:
            raise ValueError("Input dataframe must contain at least one row.")

    def _initial_temperature_field(self, *, initial_temperature: float, ambient_temperature: float) -> np.ndarray:
        if self.initialisation == "ambient":
            return np.full((self.nx, self.ny, self.nz), fill_value=initial_temperature, dtype=float)
        if self.initialisation == "linear_hot_to_cold":
            x = np.linspace(0.0, 1.0, self.nx)
            base = initial_temperature * (1.0 - x) + ambient_temperature * x
            T = np.zeros((self.nx, self.ny, self.nz), dtype=float)
            for i in range(self.nx):
                T[i, :, :] = base[i]
            return T
        raise ValueError(f"Unknown initialisation mode: {self.initialisation}")

    def _update_node_temperature(
        self,
        *,
        T: np.ndarray,
        i: int,
        j: int,
        k: int,
        thermal_conductivity: float,
        volumetric_heat_source: float,
        heat_flux_x0: float,
        convective_h: float,
        ambient_temperature: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> float:
        current = T[i, j, k]
        conduction_weight_sum = 0.0
        rhs_sum = 0.0

        def add_neighbour(ii: int, jj: int, kk: int, spacing: float) -> None:
            nonlocal conduction_weight_sum, rhs_sum
            w = thermal_conductivity / (spacing**2)
            conduction_weight_sum += w
            rhs_sum += w * T[ii, jj, kk]

        if i > 0:
            add_neighbour(i - 1, j, k, dx)
        if i < self.nx - 1:
            add_neighbour(i + 1, j, k, dx)
        if j > 0:
            add_neighbour(i, j - 1, k, dy)
        if j < self.ny - 1:
            add_neighbour(i, j + 1, k, dy)
        if k > 0:
            add_neighbour(i, j, k - 1, dz)
        if k < self.nz - 1:
            add_neighbour(i, j, k + 1, dz)

        rhs_sum += volumetric_heat_source

        if i == 0:
            w_conv = convective_h / max(dx, 1e-12)
            conduction_weight_sum += w_conv
            rhs_sum += w_conv * ambient_temperature
            rhs_sum += heat_flux_x0 / max(dx, 1e-12)
        if i == self.nx - 1:
            w_conv = convective_h / max(dx, 1e-12)
            conduction_weight_sum += w_conv
            rhs_sum += w_conv * ambient_temperature
        if j == 0 or j == self.ny - 1:
            w_conv = convective_h / max(dy, 1e-12)
            conduction_weight_sum += w_conv
            rhs_sum += w_conv * ambient_temperature
        if k == 0 or k == self.nz - 1:
            w_conv = convective_h / max(dz, 1e-12)
            conduction_weight_sum += w_conv
            rhs_sum += w_conv * ambient_temperature

        if conduction_weight_sum <= 0:
            return current
        return rhs_sum / conduction_weight_sum

    def _build_points(self, *, length_x: float | None = None, length_y: float | None = None, length_z: float | None = None) -> np.ndarray:
        lx = self.length_x if length_x is None else length_x
        ly = self.length_y if length_y is None else length_y
        lz = self.length_z if length_z is None else length_z
        xs = np.linspace(0.0, lx, self.nx)
        ys = np.linspace(0.0, ly, self.ny)
        zs = np.linspace(0.0, lz, self.nz)
        points = []
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    points.append([xs[i], ys[j], zs[k]])
        return np.asarray(points, dtype=float)

    def _node_index(self, i: int, j: int, k: int) -> int:
        return i * (self.ny * self.nz) + j * self.nz + k

    def _build_tet_cells(self) -> list[list[int]]:
        cells: list[list[int]] = []
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                for k in range(self.nz - 1):
                    n000 = self._node_index(i, j, k)
                    n100 = self._node_index(i + 1, j, k)
                    n110 = self._node_index(i + 1, j + 1, k)
                    n010 = self._node_index(i, j + 1, k)
                    n001 = self._node_index(i, j, k + 1)
                    n101 = self._node_index(i + 1, j, k + 1)
                    n111 = self._node_index(i + 1, j + 1, k + 1)
                    n011 = self._node_index(i, j + 1, k + 1)
                    # Split each structured cube into 6 tetrahedra sharing the n000-n111 body diagonal.
                    cells.extend(
                        [
                            [n000, n100, n110, n111],
                            [n000, n110, n010, n111],
                            [n000, n010, n011, n111],
                            [n000, n011, n001, n111],
                            [n000, n001, n101, n111],
                            [n000, n101, n100, n111],
                        ]
                    )
        return cells

    def _flatten_temperature_field(self, T: np.ndarray) -> np.ndarray:
        flat = np.zeros(self.nx * self.ny * self.nz, dtype=float)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    flat[self._node_index(i, j, k)] = T[i, j, k]
        return flat

    def _compute_cell_temperature_from_points(self, *, cells: list[list[int]], point_temperature: np.ndarray) -> np.ndarray:
        cell_temp = np.zeros(len(cells), dtype=float)
        for idx, cell in enumerate(cells):
            cell_temp[idx] = float(np.mean(point_temperature[np.asarray(cell)]))
        return cell_temp
