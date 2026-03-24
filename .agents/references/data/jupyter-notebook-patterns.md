# Jupyter Notebook Patterns

## When to Use

### Notebook vs Script Decision Table

| Scenario | Notebook | Script | Hybrid |
|---|---|---|---|
| Exploratory analysis, one-off | Yes | | |
| Production data pipeline | | Yes | |
| Report with inline visuals | Yes | | |
| Shared team utility | | Yes | |
| Parameterized batch runs | | | Papermill |
| ML experiment tracking | Yes | | |
| CI/CD artifact generation | | | nbconvert |
| Code review required | | Yes | nbstripout + notebook |

## Clean Diffs with nbstripout

Notebook JSON includes execution counts, cell outputs, and metadata noise. Strip it.

### Git Filter Setup

```bash
pip install nbstripout
nbstripout --install          # repo-level (.gitattributes + .git/config)
nbstripout --install --global # or globally
```

### Pre-commit hook alternative

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
        args: [--extra-keys, "metadata.kernelspec metadata.language_info"]
```

The `--extra-keys` flag strips kernel metadata that causes spurious diffs across environments.

## Parameterized Execution with Papermill

Tag a cell with `parameters` in notebook metadata. Papermill injects a new cell after it.

```python
# Cell tagged "parameters"
input_path = "data/default.csv"
n_samples = 1000
model_type = "xgboost"
output_dir = "results/"
```

### Execute Programmatically

```python
import papermill as pm

pm.execute_notebook(
    "template.ipynb",
    "output/run_2024_01.ipynb",
    parameters={"input_path": "data/january.csv", "n_samples": 5000},
    kernel_name="python3",
)
```

### Batch Execution

```python
from pathlib import Path
import papermill as pm

for month in ["jan", "feb", "mar"]:
    output = Path(f"output/{month}_report.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        "template.ipynb", str(output),
        parameters={"input_path": f"data/{month}.csv", "month": month},
        request_save_on_cell_execute=True,
    )
```

## Notebook-to-Script Conversion

### nbconvert CLI

```bash
jupyter nbconvert --to script notebook.ipynb          # .py
jupyter nbconvert --to html --execute notebook.ipynb --no-input  # report
```

### jupytext Round-Trip Sync

```bash
jupytext --set-formats ipynb,py:percent notebook.ipynb
jupytext --sync notebook.ipynb
```

## Testing Notebooks

### Execution Tests

```python
# tests/conftest.py
import subprocess, pytest
from pathlib import Path

NOTEBOOK_DIR = Path("notebooks")

@pytest.fixture(params=sorted(NOTEBOOK_DIR.glob("*.ipynb")), ids=lambda p: p.stem)
def notebook_path(request):
    return request.param

def execute_notebook(path, timeout=300):
    return subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         f"--ExecutePreprocessor.timeout={timeout}", "--output", "/dev/null", str(path)],
        capture_output=True, text=True,
    )

# tests/test_notebooks.py
def test_notebook_executes(notebook_path):
    result = execute_notebook(notebook_path)
    assert result.returncode == 0, f"{notebook_path.name} failed:\n{result.stderr[-500:]}"
```

### Targeted Cell Testing with testbook

```python
from testbook import testbook

@testbook("notebooks/preprocessing.ipynb", execute=["setup", "clean_data"])
def test_cleaning_removes_nulls(tb):
    df = tb.ref("df_clean")
    assert df.isnull().sum().sum() == 0
```

## Notebook to Production

Refactoring notebooks into production code. See `references/production-migration.md` for config management, scheduling, artifact management, and project structure.

### Refactoring Phases

| Phase | Goal | Output |
|-------|------|--------|
| 1. Assess | Understand what the notebook does | Dependency map, data flow |
| 2. Extract | Pull cells into functions/modules | Python package with clear API |
| 3. Test | Validate behavior matches notebook | Test suite with fixtures |
| 4. Configure | Externalize hardcoded values | Config files, env vars |
| 5. Schedule | Automate execution | DAG, cron, or CI pipeline |
| 6. Monitor | Track runs, data quality, drift | Logging, alerts, dashboards |

### Assessment Checklist

Before extracting code, answer: What are the inputs/outputs? Which cells are exploratory vs essential? What's the true execution order? Are there hidden cell dependencies? How often does this run? Who consumes the output?

### Notebook Anti-Patterns to Fix

| Anti-Pattern | Fix |
|---|---|
| Global mutable state (`df` across 20 cells) | Functions with explicit inputs/outputs |
| Magic numbers (`df[df['score'] > 0.73]`) | Named constants or config values |
| No error handling (silent NaN propagation) | Explicit validation, fail fast |
| Hidden cell dependencies | Explicit function call chain |
| Display-as-validation (`df.head()`) | Proper assert/test statements |
| Path hardcoding (`/Users/alice/data.csv`) | Config-driven paths |
| Mega-cell (200 lines) | Break into focused functions |
| Credential leakage in outputs | Environment variables, secrets manager |

### Module Extraction

```python
# BEFORE: notebook cell
df = pd.read_csv('sales.csv')
df = df.dropna(subset=['revenue'])
df['date'] = pd.to_datetime(df['date'])
df = df[df['revenue'] > 0]

# AFTER: extracted function
def clean_sales_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulls, parse dates, filter positive revenue."""
    df = raw_df.dropna(subset=['revenue']).copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['revenue'] > 0]
    return df
```

Key rules: explicit inputs/outputs, no global state, `.copy()` DataFrames, type hints, docstrings.

### Function-to-Module Mapping

```
cells 1-3   (load)      → src/ingestion.py
cells 4-8   (clean)     → src/preprocessing.py
cells 9-12  (features)  → src/features.py
cells 13-16 (train)     → src/training.py
cells 17-20 (evaluate)  → src/evaluation.py
```

### Pipeline Orchestration

```python
# src/pipeline.py
def run_pipeline(config: dict) -> dict:
    raw = load_sales_data(config['data_path'])
    clean = clean_sales_data(raw)
    features = build_features(clean, config['feature_params'])
    model = train_model(features, config['model_params'])
    metrics = evaluate_model(model, features)
    return {'model': model, 'metrics': metrics}
```

## Gotchas

- **Hidden state**: Running cells out of order creates unreproducible state. `Kernel > Restart & Run All` before commit. Automate with `--execute` in CI.
- **Large outputs in git**: Output cells with images/dataframes bloat history. Use nbstripout; store HTML artifacts for intentional output.
- **Kernel dependency drift**: `!pip install` inside notebooks is unreproducible. Pin deps in `pyproject.toml`. Use `%pip install` if you must install in-notebook.
- **Import scatter**: Imports across cells are hard to extract. First code cell = all imports.
- **Display side effects**: `df.head()`, `plt.show()` are not logic. Remove from production code; add logging.
- **Implicit pandas state**: `pd.set_option()` in early cells affects everything. Make explicit in config or function scope.
- **Memory assumptions**: Notebooks run on dev machines. Production may have less RAM. Profile and consider chunked processing.
- **Seed management**: Global `np.random.seed(42)` is fragile. Pass `random_state` explicitly.
- **Circular imports**: Splitting one notebook into modules often creates cycles. Draw import graph first.
- **Premature orchestration**: Get pipeline working as a single script before adding Airflow/Dagster.
