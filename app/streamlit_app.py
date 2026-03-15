from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from digilab_samplers import sample_parameter_space
from digilab_simulators.simulators import CubeThermalSteadyStateConfig, simulator_factory
from digilab_simulators.vtu import write_vtu

INDIGO = "#16425B"
KEPPEL = "#16D5C2"
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

DEFAULT_SAMPLING = {
    "method": "lhs",
    "n_samples": 10,
    "seed": 42,
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


def _sampling_from_state() -> dict[str, int | str]:
    return dict(st.session_state.get("sampling_config", DEFAULT_SAMPLING))


def _build_setup_markup(
    active_setup: dict[str, float | int | str] | None,
    sampling_config: dict[str, int | str] | None,
) -> str:
    if active_setup is None:
        return """
        <div class="hero">
            <h1 style="margin-bottom:0.2rem;">Cube Thermal Steady-State Demo</h1>
            <p style="margin:0;">
                Mock 3D thermal simulator with structured mesh output, batch sampling, nodal temperature fields, 2D slices, and VTU export.
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
        ("init", active_setup["initialisation"]),
    ]
    if sampling_config is not None:
        chips.extend(
            [
                ("sampling", sampling_config["method"]),
                ("samples", sampling_config["n_samples"]),
                ("seed", sampling_config["seed"]),
            ]
        )
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
            Mock 3D thermal simulator with structured mesh output, batch sampling, nodal temperature fields, 2D slices, and VTU export.
        </p>
        <div class="hero-grid">{chip_markup}</div>
    </div>
    """


def _build_simulator(active_setup: dict[str, float | int | str]):
    config = CubeThermalSteadyStateConfig(**active_setup)
    return config, simulator_factory(config)


@st.cache_data(show_spinner=False)
def _sample_and_evaluate(
    active_setup: dict[str, float | int | str],
    sampling_config: dict[str, int | str],
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    config, simulator = _build_simulator(active_setup)
    sample_inputs_df = sample_parameter_space(
        method=str(sampling_config["method"]),
        n_samples=int(sampling_config["n_samples"]),
        seed=int(sampling_config["seed"]),
    )
    outputs = simulator.forward(sample_inputs_df)
    field_df = pd.DataFrame(
        [output["point_data"]["temperature"] for output in outputs],
        columns=[f"node_{idx}" for idx in range(len(outputs[0]["point_data"]["temperature"]))],
    )
    return sample_inputs_df, outputs, field_df


active_setup = st.session_state.get("active_setup")
sampling_config = st.session_state.get("sampling_config")
st.markdown(_build_setup_markup(active_setup, sampling_config), unsafe_allow_html=True)

setup_expanded = "active_setup" not in st.session_state or st.session_state.get("editing_setup", False)
with st.expander("Mesh and Solver Setup", expanded=setup_expanded):
    with st.form("setup_form"):
        current_setup = _setup_from_state()

        row1 = st.columns(5)
        nx = row1[0].slider("nx", 4, 16, int(current_setup["nx"]))
        ny = row1[1].slider("ny", 4, 16, int(current_setup["ny"]))
        nz = row1[2].slider("nz", 4, 16, int(current_setup["nz"]))
        max_iterations = row1[3].slider("max_iterations", 50, 4000, int(current_setup["max_iterations"]), step=50)
        initialisation = row1[4].selectbox(
            "initialisation",
            ["ambient", "linear_hot_to_cold"],
            index=["ambient", "linear_hot_to_cold"].index(str(current_setup["initialisation"])),
        )

        row2 = st.columns(4)
        length_x = row2[0].number_input("length_x", value=float(current_setup["length_x"]), min_value=0.1)
        length_y = row2[1].number_input("length_y", value=float(current_setup["length_y"]), min_value=0.1)
        length_z = row2[2].number_input("length_z", value=float(current_setup["length_z"]), min_value=0.1)
        ambient_temperature = row2[3].number_input(
            "ambient_temperature [K]",
            value=float(current_setup["ambient_temperature"]),
        )

        row3 = st.columns(4)
        tolerance = row3[0].number_input(
            "tolerance",
            value=float(current_setup["tolerance"]),
            format="%.1e",
        )
        row3[1].empty()
        row3[2].empty()
        row3[3].empty()

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
        st.session_state.pop("sampling_config", None)
        st.session_state["editing_setup"] = False
        st.rerun()

if "active_setup" not in st.session_state:
    st.info("Run setup to build the mesh and enable sampling.")
    st.stop()

change_setup_col, _ = st.columns([0.22, 0.78])
with change_setup_col:
    if st.button("Change setup", use_container_width=True):
        st.session_state["editing_setup"] = True
        st.session_state.pop("sampling_config", None)
        st.rerun()

active_setup = st.session_state["active_setup"]
config, simulator = _build_simulator(active_setup)

nodes_df = simulator.mesh_nodes_to_dataframe(
    length_x=config.length_x,
    length_y=config.length_y,
    length_z=config.length_z,
)
elements_df = simulator.mesh_elements_to_dataframe()

st.subheader("Mesh preview")
preview_col1, preview_col2 = st.columns(2)
with preview_col1:
    st.caption("Nodes")
    st.dataframe(nodes_df.head(8), use_container_width=True, height=220)
    st.download_button(
        "Download nodes CSV",
        data=nodes_df.to_csv(index=False).encode("utf-8"),
        file_name="cube_mesh_nodes.csv",
        mime="text/csv",
        use_container_width=True,
    )
with preview_col2:
    st.caption("Elements")
    st.dataframe(elements_df.head(8), use_container_width=True, height=220)
    st.download_button(
        "Download elements CSV",
        data=elements_df.to_csv(index=False).encode("utf-8"),
        file_name="cube_mesh_elements.csv",
        mime="text/csv",
        use_container_width=True,
    )

sampling_expanded = "sampling_config" not in st.session_state
with st.expander("Sampling", expanded=sampling_expanded):
    with st.form("sampling_form"):
        current_sampling = _sampling_from_state()
        sampling_cols = st.columns(3)
        sampling_method = sampling_cols[0].selectbox(
            "sampling_method",
            ["lhs", "sobol"],
            index=["lhs", "sobol"].index(str(current_sampling["method"])),
        )
        n_samples = sampling_cols[1].slider("n_samples", 2, 32, int(current_sampling["n_samples"]))
        seed = sampling_cols[2].number_input("seed", min_value=0, value=int(current_sampling["seed"]), step=1)
        run_sampling = st.form_submit_button("Run sampling", use_container_width=True)

    if run_sampling:
        st.session_state["sampling_config"] = {
            "method": sampling_method,
            "n_samples": int(n_samples),
            "seed": int(seed),
        }
        st.rerun()

if "sampling_config" not in st.session_state:
    st.info("Choose a sampling method and sample count, then run sampling to generate the input and field-data tables.")
    st.stop()

sampling_config = st.session_state["sampling_config"]
sample_inputs_df, outputs, field_df = _sample_and_evaluate(active_setup, sampling_config)

st.subheader("Sampled data")
st.caption(
    f"{len(sample_inputs_df)} samples generated with {sampling_config['method'].upper()} on the active mesh."
)
data_left, data_right = st.columns(2)
with data_left:
    st.caption("Inputs")
    st.dataframe(sample_inputs_df, use_container_width=True, height=320)
    st.download_button(
        "Download inputs CSV",
        data=sample_inputs_df.to_csv(index=False).encode("utf-8"),
        file_name="cube_thermal_sample_inputs.csv",
        mime="text/csv",
        use_container_width=True,
    )
with data_right:
    st.caption("Field data: nodal temperature")
    st.dataframe(field_df, use_container_width=True, height=320)
    st.download_button(
        "Download field data CSV",
        data=field_df.to_csv(index=False).encode("utf-8"),
        file_name="cube_thermal_field_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

selected_sample = st.slider("Sample index", 0, len(outputs) - 1, 0)
selected_output = outputs[selected_sample]

temperature = np.array(selected_output["point_data"]["temperature"]).reshape(config.nx, config.ny, config.nz)
mid_k = config.nz // 2
slice_xy = temperature[:, :, mid_k]

fig1, ax1 = plt.subplots(figsize=(6, 4.5))
im = ax1.imshow(slice_xy.T, origin="lower", aspect="auto", cmap="inferno")
ax1.set_xlabel("x node index")
ax1.set_ylabel("y node index")
ax1.set_title(f"Mid-plane temperature slice for sample {selected_sample}")
fig1.colorbar(im, ax=ax1, label="Temperature [K]")
st.pyplot(fig1)

st.subheader("VTU export")
with tempfile.TemporaryDirectory() as tmpdir:
    out_path = Path(tmpdir) / f"cube_thermal_output_sample_{selected_sample}.vtu"
    write_vtu(selected_output, out_path)
    st.download_button(
        "Download selected sample VTU",
        data=out_path.read_bytes(),
        file_name=out_path.name,
        mime="application/octet-stream",
    )
