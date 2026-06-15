from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from path_utils import BASE_DIR
from run_observed_pareto import analyze_observed_dataframe
from run_surrogate_nsga2 import run_surrogate_analysis


st.set_page_config(page_title="HVAC Dataset Pareto & NSGA-II", layout="wide")
st.title("HVAC Dataset Pareto and Surrogate-Assisted NSGA-II")
st.caption(
    "Use solver-generated CSV outputs for observed Pareto ranking and exploratory surrogate-assisted policy optimization."
)

bundled_dataset = BASE_DIR / "data" / "matrix_ml_dataset.csv"

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload matrix_ml_dataset.csv", type=["csv"])
use_bundled = st.sidebar.checkbox(
    "Use bundled dataset",
    value=uploaded is None and bundled_dataset.is_file(),
    disabled=not bundled_dataset.is_file(),
)


def load_dataset() -> tuple[pd.DataFrame | None, str]:
    if uploaded is not None:
        return pd.read_csv(uploaded), uploaded.name
    if use_bundled and bundled_dataset.is_file():
        return pd.read_csv(bundled_dataset), str(bundled_dataset)
    return None, ""


try:
    dataframe, source_label = load_dataset()
except Exception as exc:
    st.error(f"The CSV could not be read: {exc}")
    st.stop()

if dataframe is None:
    st.info(
        "Upload the solver CSV in the sidebar. The command-line analysis scripts are not Streamlit apps; "
        "for Streamlit Cloud, set the Main file path to streamlit_app.py."
    )
    st.stop()

st.success(f"Loaded {len(dataframe):,} rows from {source_label}")
with st.expander("Dataset columns and preview"):
    st.write(dataframe.columns.tolist())
    st.dataframe(dataframe.head(20), use_container_width=True)

observed_tab, optimization_tab, notes_tab = st.tabs([
    "Observed Pareto analysis",
    "Surrogate NSGA-II",
    "Scientific interpretation",
])

with observed_tab:
    st.subheader("Exact post-processing of existing scenarios")
    if st.button("Run observed Pareto analysis", type="primary"):
        try:
            summary, pareto = analyze_observed_dataframe(dataframe)
            st.session_state["observed_summary"] = summary
            st.session_state["observed_pareto"] = pareto
        except Exception as exc:
            st.exception(exc)

    if "observed_summary" in st.session_state:
        summary = st.session_state["observed_summary"]
        pareto = st.session_state["observed_pareto"]
        st.markdown("#### Scenario summary")
        st.dataframe(summary, use_container_width=True)
        st.markdown("#### Nondominated scenarios")
        st.dataframe(pareto, use_container_width=True)
        st.download_button(
            "Download observed scenario summary",
            summary.to_csv(index=False).encode("utf-8"),
            "observed_scenario_summary.csv",
            "text/csv",
        )
        st.download_button(
            "Download observed Pareto solutions",
            pareto.to_csv(index=False).encode("utf-8"),
            "observed_pareto_solutions.csv",
            "text/csv",
        )

with optimization_tab:
    st.subheader("Exploratory surrogate-assisted NSGA-II")
    st.warning(
        "This optimization does not rerun the physics solver. The selected policy is a candidate that must be validated "
        "with the corrected core solver before publication."
    )

    mode = st.radio("Run mode", ["Quick pilot", "Custom"], horizontal=True)
    if mode == "Quick pilot":
        trees, contexts, population, generations = 100, 200, 20, 10
    else:
        col1, col2, col3, col4 = st.columns(4)
        trees = col1.number_input("Trees", 50, 600, 250, 50)
        contexts = col2.number_input("Context rows/severity (0 = all)", 0, 5000, 500, 100)
        population = col3.number_input("Population", 10, 150, 32, 2)
        generations = col4.number_input("Generations", 5, 300, 30, 5)

    if st.button("Run surrogate NSGA-II", type="primary"):
        try:
            with st.spinner("Training surrogate models and running NSGA-II..."):
                temp_root = Path(tempfile.mkdtemp(prefix="hvac_nsga2_"))
                input_path = temp_root / "uploaded_dataset.csv"
                output_path = temp_root / "outputs"
                config_path = BASE_DIR / "config.json"
                dataframe.to_csv(input_path, index=False)
                result = run_surrogate_analysis(
                    input_path=input_path,
                    config_path=config_path,
                    output_dir=output_path,
                    config_overrides={
                        "trees": int(trees),
                        "context_rows_per_severity": int(contexts),
                        "population_size": int(population),
                        "generations": int(generations),
                    },
                )
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    for file_path in output_path.rglob("*"):
                        if file_path.is_file():
                            archive.write(file_path, file_path.relative_to(output_path))
                zip_buffer.seek(0)
                st.session_state["surrogate_result"] = result
                st.session_state["surrogate_zip"] = zip_buffer.getvalue()
        except Exception as exc:
            st.exception(exc)

    if "surrogate_result" in st.session_state:
        result = st.session_state["surrogate_result"]
        st.markdown("#### Dataset audit")
        st.json(result["audit"])
        st.markdown("#### Surrogate validation metrics")
        st.dataframe(result["metrics"], use_container_width=True)
        st.markdown("#### Selected closest-to-ideal candidate policy")
        st.dataframe(result["selected_policy"], use_container_width=True)
        st.markdown("#### Pareto candidate policies")
        st.dataframe(result["pareto"], use_container_width=True)
        st.download_button(
            "Download all optimization outputs (ZIP)",
            st.session_state["surrogate_zip"],
            "hvac_surrogate_nsga2_outputs.zip",
            "application/zip",
        )

with notes_tab:
    st.markdown(
        """
### Correct interpretation

- **Observed Pareto analysis** ranks only the strategy–severity scenarios already present in the CSV.
- **Surrogate NSGA-II** searches a fitted response surface and creates candidate setpoint/airflow policies.
- It does not update fouling, dust, pressure drop, or maintenance timing dynamically under the new policy.
- Final energy savings must come from a fresh run of the corrected physics solver, not from the surrogate prediction alone.

### Streamlit Cloud setup

Set the application **Main file path** to `streamlit_app.py`. Do not select `run_observed_pareto.py` or `run_surrogate_nsga2.py`; those are command-line scripts.
        """
    )
