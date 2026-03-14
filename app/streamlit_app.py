from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from digilab_simulators.simulators import CubeThermalSteadyStateConfig, simulator_factory
from digilab_simulators.vtu import write_vtu

INDIGO = "#16425B"
KEPPEL = "#16D5C2"
KEY_LIME = "#EBF38B"
LIGHT_BG = "#F7FAFC"

DEFAULT_SETUP = {
    "nx": 8,
    "ny": 8,
    "nz": 8,
    "length_x": 1.0,
    "length_y": 0.75,
    "length_z": 0.5,
    "ambient_temperature": 293.15,
    "max_iterations": 1500,
    "tolerance": 1e-5,
    "initialisation": "linear_hot_to_cold",
}

st.set_page_config(page_title="Cube Thermal Demo", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {LIGHT_BG}; }}
    .hero {{
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        background: linear-gradient(90deg, {INDIGO}, {KEPPEL});
        color: white;
        margin-bottom: 1rem;
    }}
    .hero-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.65rem;
        margin-top: 0.9rem;
    }}
    .hero-chip {{
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 0.8rem;
        padding: 0.55rem 0.7rem;
    }}
    .hero-chip-label {{
        font-size: 0.75rem;
        opacity: 0.85;
    }}
    .hero-chip-value {{
        font-size: 1rem;
        font-weight: 700;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _setup_from_state() -> dict[str, float | int | str]:
    return dict(st.session_state.get("active_setup", DEFAULT_SETUP))


def _build_setup_markup(active_setup: dict[str, float | int | str] | None) -> str:
    if active_setup is None:
        return """
        <div class="hero">
            <h1 style="margin-bottom:0.2rem;">Cube Thermal Steady-State Demo</h1>
            <p style="margin:0;">
                Mock 3D thermal simulator with structured mesh output, summary metrics, 2D slices, and VTU export.
            </p>
            <p style="margin:0.9rem 0 0;">
                No setup has been run yet. Configure the mesh and solver below, then press <strong>Run setup</strong>.
            </p>
        </div>
        """

    chips = [
        ("nx", active_setup["nx"]),
        ("ny", active_setup["ny"]),
        ("nz", active_setup["nz"]),
        ("length_x", active_setup["length_x"]),
        ("length_y", active_setup["length_y"]),
        ("length_z", active_setup["length_z"]),
        ("ambient [K]", active_setup["ambient_temperature"]),
        ("max_iterations", active_setup["max_iterations"]),
        ("tolerance", active_setup["tolerance"]),
        ("initialisation", active_setup["initialisation"]),
    ]
    chip_markup = "".join(
        f"""
        <div class="hero-chip">
            <div class="hero-chip-label">{label}</div>
            <div class="hero-chip-value">{value}</div>
        </div>
        """
        for label, value in chips
    )
    return f"""
    <div class="hero">
        <h1 style="margin-bottom:0.2rem;">Cube Thermal Steady-State Demo</h1>
        <p style="margin:0;">
            Mock 3D thermal simulator with structured mesh output, summary metrics, 2D slices, and VTU export.
        </p>
        <div class="hero-grid">{chip_markup}</div>
    </div>
    """


def _build_simulator(active_setup: dict[str, float | int | str]):
    config = CubeThermalSteadyStateConfig(**active_setup)
    return config, simulator_factory(config)


active_setup = st.session_state.get("active_setup")
st.markdown(_build_setup_markup(active_setup), unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Mesh and Solver Setup")
    with st.form("setup_form"):
        row1 = st.columns(5)
        nx = row1[0].slider("nx", 4, 16, int(_setup_from_state()["nx"]))
        ny = row1[1].slider("ny", 4, 16, int(_setup_from_state()["ny"]))
        nz = row1[2].slider("nz", 4, 16, int(_setup_from_state()["nz"]))
        max_iterations = row1[3].slider("max_iterations", 50, 4000, int(_setup_from_state()["max_iterations"]), step=50)
        initialisation = row1[4].selectbox(
            "initialisation",
            ["ambient", "linear_hot_to_cold"],
            index=["ambient", "linear_hot_to_cold"].index(str(_setup_from_state()["initialisation"])),
        )

        row2 = st.columns(4)
        length_x = row2[0].number_input("length_x", value=float(_setup_from_state()["length_x"]), min_value=0.1)
        length_y = row2[1].number_input("length_y", value=float(_setup_from_state()["length_y"]), min_value=0.1)
        length_z = row2[2].number_input("length_z", value=float(_setup_from_state()["length_z"]), min_value=0.1)
        ambient_temperature = row2[3].number_input(
            "ambient_temperature [K]",
            value=float(_setup_from_state()["ambient_temperature"]),
        )

        tolerance = st.number_input(
            "tolerance",
            value=float(_setup_from_state()["tolerance"]),
            format="%.1e",
        )
        submitted = st.form_submit_button("Run setup", use_container_width=True)

    if submitted:
        st.session_state["active_setup"] = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "length_x": float(length_x),
            "length_y": float(length_y),
            "length_z": float(length_z),
            "ambient_temperature": float(ambient_temperature),
            "max_iterations": max_iterations,
            "tolerance": float(tolerance),
            "initialisation": initialisation,
        }
        st.rerun()

if "active_setup" not in st.session_state:
    st.info("Run setup to build the mesh and enable downloads and solver results.")
    st.stop()

active_setup = st.session_state["active_setup"]
config, simulator = _build_simulator(active_setup)
nodes_df = simulator.mesh_nodes_to_dataframe(
    length_x=config.length_x,
    length_y=config.length_y,
    length_z=config.length_z,
)
elements_df = simulator.mesh_elements_to_dataframe()

mesh_left, mesh_right = st.columns([1.2, 1])
with mesh_left:
    st.subheader("Mesh exports")
    st.caption(
        f"{len(nodes_df)} nodes and {len(elements_df)} tetrahedral elements are available from the active setup."
    )
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download nodes CSV",
        data=nodes_df.to_csv(index=False).encode("utf-8"),
        file_name="cube_mesh_nodes.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_right.download_button(
        "Download elements CSV",
        data=elements_df.to_csv(index=False).encode("utf-8"),
        file_name="cube_mesh_elements.csv",
        mime="text/csv",
        use_container_width=True,
    )

with mesh_right:
    st.subheader("Mesh preview")
    preview_col1, preview_col2 = st.columns(2)
    preview_col1.dataframe(nodes_df.head(10), use_container_width=True)
    preview_col2.dataframe(elements_df.head(10), use_container_width=True)

st.subheader("Inputs")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    thermal_conductivity = st.number_input("thermal_conductivity [W/m/K]", value=12.0, min_value=0.1)
with col2:
    volumetric_heat_source = st.number_input("volumetric_heat_source [W/m³]", value=2.0e4, format="%.3e")
with col3:
    heat_flux_x0 = st.number_input("heat_flux_x0 [W/m²]", value=1.5e4, format="%.3e")
with col4:
    convective_h = st.number_input("convective_h [W/m²/K]", value=8.0, min_value=0.0)
with col5:
    initial_temperature = st.number_input("initial_temperature [K]", value=293.15)

X = pd.DataFrame(
    [
        {
            "thermal_conductivity": thermal_conductivity,
            "volumetric_heat_source": volumetric_heat_source,
            "heat_flux_x0": heat_flux_x0,
            "convective_h": convective_h,
            "initial_temperature": initial_temperature,
        }
    ]
)

output = simulator.forward(X)[0]
summary_df = simulator.outputs_to_summary_dataframe([output])
point_df = simulator.output_to_point_dataframe(output)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Centre temperature [K]", f"{output['centre_temperature']:.2f}")
m2.metric("Temperature range [K]", f"{output['temperature_range']:.2f}")
m3.metric("Iterations", output["iterations"])
m4.metric("Converged", "Yes" if output["converged"] else "No")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True)
    st.download_button(
        "Download summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="cube_thermal_summary.csv",
        mime="text/csv",
    )

with right:
    st.subheader("Point data preview")
    st.dataframe(point_df.head(20), use_container_width=True)

temperature = np.array(output["point_data"]["temperature"]).reshape(config.nx, config.ny, config.nz)
mid_k = config.nz // 2
slice_xy = temperature[:, :, mid_k]

fig1, ax1 = plt.subplots(figsize=(6, 4.5))
im = ax1.imshow(slice_xy.T, origin="lower", aspect="auto", cmap="inferno")
ax1.set_xlabel("x node index")
ax1.set_ylabel("y node index")
ax1.set_title("Mid-plane temperature slice")
fig1.colorbar(im, ax=ax1, label="Temperature [K]")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.scatter(point_df["x"], point_df["temperature"], s=18)
ax2.set_xlabel("x [m]")
ax2.set_ylabel("Temperature [K]")
ax2.set_title("Point temperature vs x")
st.pyplot(fig2)

st.subheader("VTU export")
with tempfile.TemporaryDirectory() as tmpdir:
    out_path = Path(tmpdir) / "cube_thermal_output.vtu"
    write_vtu(output, out_path)
    st.download_button(
        "Download VTU",
        data=out_path.read_bytes(),
        file_name="cube_thermal_output.vtu",
        mime="application/octet-stream",
    )
