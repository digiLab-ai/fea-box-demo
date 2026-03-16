# Cube Thermal Steady-State Demo

A self-contained demo repository for a mock 3D cube thermal steady-state simulator, bundled with:

- a Streamlit app
- a notebook
- pytest coverage for the simulator
- a lightweight VTU exporter for ParaView

## Simulator Dynamics

This simulator models steady-state heat conduction in a 3D rectangular cube using a finite difference method on a structured grid. The governing equation is the heat conduction equation in steady-state:

∇·(k ∇T) + q = 0

where:
- **T** is the temperature field (K)
- **k** is the thermal conductivity (W/m·K), assumed constant
- **q** is the volumetric heat source (W/m³), representing internal heat generation per unit volume

The equation is discretized using finite differences, resulting in a system of linear equations solved iteratively until convergence.

### Boundary Conditions
The cube has convective heat transfer on all faces to the ambient environment at temperature T_amb, except the x=0 face which has convection to an external heat source at temperature T_source.

- **x=0 face**: Convection to external source: -k ∂T/∂x = h (T_source - T)
- **x=Lx face**: Convection to ambient: -k ∂T/∂x = h (T - T_amb)
- **y=0 and y=Ly faces**: Convection to ambient: -k ∂T/∂y = h (T - T_amb)
- **z=0 and z=Lz faces**: Convection to ambient: -k ∂T/∂z = h (T - T_amb)

where **h** is the convective heat transfer coefficient (W/m²·K).

### Input Parameters
The simulator takes the following input parameters for each simulation sample:

- **thermal_conductivity** (W/m·K): Thermal conductivity of the cube material. Typical range: 5.0 - 25.0. Determines how easily heat conducts through the material.
- **volumetric_heat_source** (W/m³): Internal heat generation rate per unit volume. Typical range: 5.0e3 - 4.0e4. Represents sources like electrical heating or chemical reactions.
- **T_source** (K): Temperature of the external heat source at the x=0 face. Typical range: 300.0 - 1000.0. Drives heat into the cube via convection.
- **convective_h** (W/m²·K): Convective heat transfer coefficient on all boundaries. Typical range: 2.0 - 20.0. Higher values mean better heat transfer.
- **initial_temperature** (K): Initial temperature guess for the iterative solver. Typical range: 285.15 - 315.15. Affects convergence speed but not the final solution.

### Mesh and Solver Parameters
These are configuration parameters set once per simulator instance:

- **nx, ny, nz**: Number of nodes in x, y, z directions (integer, ≥2). Typical: 4-16. Higher values increase accuracy but computational cost.
- **length_x, length_y, length_z** (m): Cube dimensions. Default: 1.0, 0.75, 0.5. Physical size of the domain.
- **ambient_temperature** (K): Ambient/reference temperature. Default: 293.15 (20°C). Used in boundary conditions.
- **max_iterations**: Maximum number of solver iterations. Default: 4000. Prevents infinite loops.
- **tolerance**: Convergence tolerance on maximum nodal temperature change. Default: 1e-6. Smaller values mean more accurate solutions.
- **initialisation**: Initial temperature field guess ("ambient" or "linear_hot_to_cold"). Default: "ambient". Affects convergence speed.

## Repo structure

```text
cube-thermal-demo-repo/
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── cube_thermal_demo.ipynb
├── src/
│   └── digilab_simulators/
│       ├── __init__.py
│       ├── vtu.py
│       └── simulators/
│           ├── __init__.py
│           ├── base.py
│           └── fea_box.py
├── tests/
│   └── test_cube_thermal_simulator.py
├── pyproject.toml
└── README.md
```

## Quick start

```bash
poetry install
```

Run tests:

```bash
pytest
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Open the notebook:

```bash
jupyter notebook notebooks/cube_thermal_demo.ipynb
```

## Notes

This is a compact mock solver intended for demo workflows, surrogate modelling, and visualisation pipelines. It is not intended for engineering sign-off or certified thermal analysis.
