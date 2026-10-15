# Putting GlucoTwin online (free)

Judges are far more likely to try a link than to install anything. Two free options follow. Both serve the
dashboard straight from your public GitHub repository.

## Before deploying: commit the pre-built artefacts

The dashboard reads these files, so they must be in the repo. They are produced by `python scripts/build_all.py`,
`python scripts/hypo_real_data.py` and `python scripts/make_docs.py`, and are not git-ignored.

```
data/raw/                     ehr.csv, wearables.parquet, ground_truth.parquet, true_params.csv
data/processed/               predictions.parquet, splits.csv
models/                       glucotwin_bundle.joblib, twin_params.csv, hypo_model.joblib
reports/                      metrics.json, real_data_metrics.json, hypo_metrics.json, figures/
```

`data/processed/features_*.parquet` and `data/external/` are deliberately git-ignored. The feature tables are large
and rebuildable; the real dataset is fetched with `scripts/download_real_data.py`.

## Option A: Streamlit Community Cloud (easiest)

1. Push the repository to GitHub and make it **public**.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and choose **Create app → Deploy a public app from
   GitHub**.
3. Repository: `maxlovesher/twin_KIIT` · Branch: `main` · Main file path: **`dashboard/app.py`**.
4. Under *Advanced settings*, choose **Python 3.11**.
5. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes. The first page load
   takes ~20 s while the patient twins load.
6. Copy the app URL into the README (`▶ Live demo:` line) and onto the title slide.

**If the app runs out of memory on the free tier,** use option B.

## Option B: Hugging Face Spaces (Docker)

1. Create a new Space at <https://huggingface.co/new-space> and choose the **Docker** SDK (blank template).
2. Push this repository to the Space's git remote, including the `Dockerfile`.
3. Add this header at the very top of the Space's `README.md`, so the Space knows which port to open:

   ```yaml
   ---
   title: GlucoTwin
   sdk: docker
   app_port: 8501
   ---
   ```

4. The Space builds the image from `Dockerfile` and serves the dashboard.

## Running locally (for reviewers)

The easiest way is to double-click `start.bat` on Windows, or run `bash start.sh` on macOS or Linux. The script
creates `.venv`, installs the pinned requirements and opens the dashboard. To do the same by hand:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py                 # http://localhost:8501
uvicorn glucotwin.api:app --port 8000          # http://localhost:8000/docs
# or, with Docker Desktop running:
docker compose up --build
```

Deep links open a specific patient and moment, which is handy in the video and README:
`http://localhost:8501/?pid=GT0014&t=2026-08-12T06:45`
