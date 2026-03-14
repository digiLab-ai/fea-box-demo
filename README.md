# Cube Thermal Steady-State Demo

A self-contained demo repository for a mock 3D cube thermal steady-state simulator, bundled with:

- a Streamlit app
- a notebook
- pytest coverage for the simulator
- a lightweight VTU exporter for ParaView

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
