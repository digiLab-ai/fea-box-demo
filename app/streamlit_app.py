from __future__ import annotations

from io import BytesIO
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
    .metric-card {{
        background: white;
        border-left: 6px solid {INDIGO};
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 0.1rem 0.6rem rgba(0,0,0,0.07);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.2rem;">Cube Thermal Steady-State Demo</h1>
        <p style="margin:0;">Mock 3D thermal simulator with structured mesh output, summary metrics, 2D slices, and VTU export.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Mesh and solver")
    nx = st.slider("nx", 4, 16, 8)
    ny = st.slider("ny", 4, 16, 8)
    nz = st.slider("nz", 4, 16, 8)
    length_x = st.number_input("length_x", value=1.0, min_value=0.1)
    length_y = st.number_input("length_y", value=0.75, min_value=0.1)
    length_z = st.number_input("length_z", value=0.5, min_value=0.1)
    ambient_temperature = st.number_input("ambient_temperature [K]", value=293.15)
    max_iterations = st.slider("max_iterations", 50, 4000, 1500, step=50)
    tolerance = st.number_input("tolerance", value=1e-5, format="%.1e")
    initialisation = st.selectbox("initialisation", ["ambient", "linear_hot_to_cold"], index=1)

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

config = CubeThermalSteadyStateConfig(
    nx=nx,
    ny=ny,
    nz=nz,
    length_x=length_x,
    length_y=length_y,
    length_z=length_z,
    ambient_temperature=ambient_temperature,
    max_iterations=max_iterations,
    tolerance=tolerance,
    initialisation=initialisation,
)
simulator = simulator_factory(config)

X = pd.DataFrame([
    {
        "thermal_conductivity": thermal_conductivity,
        "volumetric_heat_source": volumetric_heat_source,
        "heat_flux_x0": heat_flux_x0,
        "convective_h": convective_h,
        "initial_temperature": initial_temperature,
    }
])

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

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download summary CSV", data=csv_bytes, file_name="cube_thermal_summary.csv", mime="text/csv")

with right:
    st.subheader("Point data preview")
    st.dataframe(point_df.head(20), use_container_width=True)

sample_output = output
temperature = np.array(sample_output["point_data"]["temperature"]).reshape(nx, ny, nz)
mid_k = nz // 2
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
