# IdeaboxAI-Vertical-ML-Service

## Setup Instruction

**1. Clone the repository**
```bash
git clone git@github.com:ideaboxai/ideaboxai-vertical-ml-service.git
cd ideaboxai-vertical-ml-service
```

**2. Set up Python Environment with uv**
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies from pyproject.toml
uv sync

# Install with development tools
uv sync --group dev
```

**3. Configure dvc (for data & model versioning)**
If you installed dev dependencies, enable auto-staging for tracked files:
```bash
dvc config core.autostage true
```
Pull the latest data and model files:
```bash
dvc pull
```

**4. Setup Pre-commit Hooks**
```bash
pre-commit install
```

## 📁 Project Structure
```bash
ideaboxai-vertical-ml-service
├── data
│  └── sandhya-aqua-erp
│     ├── output
│     ├── processed
│     └── raw
├── models
│  └── sandhya-aqua-erp
├── notebooks
│  └── anomaly_detecton
│     ├── 1_eda.ipynb
│     └── 2_model_prototyping.ipynb
├── scripts
├── src
│  ├── anomaly_detection
│  │  ├── components
│  │  │  ├── model_evaluator.py
│  │  │  └── model_trainer.py
│  │  ├── strategies
│  │  │  ├── base.py
│  │  │  ├── ml_detectors.py
│  │  │  └── stat_detectors.py
│  │  ├── utils
│  │  └── factory.py
│  └── recommendation
├── verticals
│  └── sandhya-aqua-erp
│     ├── anomaly_detection
│     │  └── supply_chain
│     │     ├── cross_stage_anomaly_detection.py
│     │     ├── feature_wise_anomaly_detection.py
│     │     └── stage_wise_anomaly_detection.py
│     ├── data_preparation
│     │  ├── farmer
│     │  └── supply_chain
│     │     ├── cross_stage_data_preparation.py
│     │     ├── individual_table_wise_data_preparation.py
│     │     └── stage_wise_data_preparation.py
│     └── docs
│        └── anomaly_detection_arch.excalidraw
├── .dvcignore
├── .env
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── data.dvc
├── models.dvc
├── pyproject.toml
└── uv.lock
```

## 📦 Dependency Management
We use uv for fast, reproducible installs.
- Prod only: `uv sync`
- Dev environment: `uv sync --group dev`
- Add new dependency: `uv add package_name`
- Add new dev dependency: `uv add --group dev package_name`

## 📊 Data Version Control (DVC)
- Track a file:
```bash
dvc add data/raw/data.csv
```

- Commit changes:

```bash
git add data/.gitignore data/raw/data.csv.dvc
git commit -m "Add raw data"
```

- Push to remote storage:
```bash
dvc push
```
