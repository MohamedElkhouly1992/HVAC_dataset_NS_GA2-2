# HVAC Dataset Pareto and Surrogate NSGA-II — Streamlit-Safe Revision

## Why the FileNotFoundError occurred

The original command-line script used the relative path `data/matrix_ml_dataset.csv`. On a local terminal this often works because the terminal is opened inside the project folder. On Streamlit Cloud, the process working directory can differ from the script directory, so the same relative path can point to a nonexistent location.

There was a second deployment issue: `run_observed_pareto.py` is a command-line script, not a Streamlit interface. The Streamlit Cloud main file must be `streamlit_app.py`.

## Corrections in this revision

- All command-line paths are resolved relative to the script/bundle directory, not only the process working directory.
- Missing inputs produce a readable error listing every searched location.
- A Streamlit application was added with a CSV uploader.
- The Streamlit app can use the bundled CSV when it exists, but does not require it.
- Uploaded datasets are handled in a temporary runtime directory.
- The Windows and Linux run scripts first change into the bundle directory.
- Baseline `NaN` maintenance-action values are interpreted as zero during observed post-processing.

## Streamlit Cloud deployment

1. Upload the complete folder to a GitHub repository.
2. Create a Streamlit Cloud app from that repository.
3. Set **Main file path** to:

```text
streamlit_app.py
```

4. Do not select `run_observed_pareto.py` or `run_surrogate_nsga2.py` as the main file.
5. The CSV can either be committed at:

```text
data/matrix_ml_dataset.csv
```

or uploaded through the application sidebar after deployment.

## Local Streamlit run

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Command-line use

The scripts now work even when called from a different working directory:

```bash
python /path/to/bundle/run_observed_pareto.py
python /path/to/bundle/run_surrogate_nsga2.py
```

An explicit input may also be passed:

```bash
python run_observed_pareto.py --input /absolute/path/to/matrix_ml_dataset.csv
```

## Scientific limitation

The observed analysis is exact post-processing of existing solver outputs. The surrogate NSGA-II analysis is exploratory: it does not dynamically rerun the physical degradation and maintenance states. Validate every selected candidate policy using the corrected core solver before using its numerical savings in a manuscript.
