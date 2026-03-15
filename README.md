# Cube Thermal Steady-State Demo

A self-contained demo repository for a mock 3D cube thermal steady-state simulator, bundled with:

- a Streamlit app
- a notebook
- pytest coverage for the simulator
- a lightweight VTU exporter for ParaView

## Simulator Dynamics

This simulator models steady-state heat conduction in a 3D rectangular cube using a finite difference method. The governing equation is the heat conduction equation:

∇·(k ∇T) + q = 0

where:
- T is the temperature field (K)
- k is the thermal conductivity (W/m·K)
- q is the volumetric heat source (W/m³)

### Boundary Conditions
- **x=0 face**: Prescribed heat flux (q_flux) + convection to ambient
- **x=Lx face**: Convection to ambient
- **y=0 and y=Ly faces**: Convection to ambient
- **z=0 and z=Lz faces**: Convection to ambient

The convective boundary condition is: -k ∂T/∂n = h (T - T_amb)

### Input Parameters
- **thermal_conductivity** (5.0 - 25.0 W/m·K): Thermal conductivity of the cube material
- **volumetric_heat_source** (5.0e3 - 4.0e4 W/m³): Internal heat generation rate per unit volume
- **heat_flux_x0** (5.0e3 - 2.5e4 W/m²): Heat flux applied to the x=0 face
- **convective_h** (2.0 - 20.0 W/m²·K): Convective heat transfer coefficient on all boundaries
- **initial_temperature** (285.15 - 315.15 K): Initial temperature guess for the iterative solver

### Mesh and Solver Parameters
- **nx, ny, nz**: Number of nodes in x, y, z directions (4-16)
- **length_x, length_y, length_z**: Cube dimensions in meters (default 1.0, 0.75, 0.5)
- **ambient_temperature**: Ambient/reference temperature (K, default 293.15)
- **max_iterations**: Maximum solver iterations (50-4000)
- **tolerance**: Convergence tolerance on max nodal temperature change (default 1e-5)
- **initialisation**: Initial field guess ("ambient" or "linear_hot_to_cold")

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
