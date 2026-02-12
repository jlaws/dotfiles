---
name: python-packaging-and-distribution
description: "Use when configuring pyproject.toml, choosing build backends, publishing to PyPI, creating conda recipes, or managing package versions."
---

# Python Packaging and Distribution

## Build Backend Decision Table

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **hatchling** | Most projects | Fast, modern, good defaults | Newer, smaller ecosystem |
| setuptools | Legacy/complex | Universal support, battle-tested | Verbose config, slow build |
| flit-core | Pure Python libs | Minimal config | No C extensions, limited |
| maturin | Rust extensions | Rust+Python seamless | Rust-only extensions |
| pdm-backend | PDM users | PEP 621, good with pdm | Tied to pdm ecosystem |

**Recommendation**: Use hatchling unless you have Rust (maturin) or need backward compat with existing setuptools config.

## pyproject.toml Complete Template

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "my-package"
dynamic = ["version"]
description = "One-line description of the package"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Your Name", email = "you@example.com" },
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Typing :: Typed",
]
dependencies = [
    "httpx>=0.25",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
]
docs = [
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.25",
]

[project.urls]
Homepage = "https://github.com/user/my-package"
Documentation = "https://my-package.readthedocs.io"
Repository = "https://github.com/user/my-package"
Issues = "https://github.com/user/my-package/issues"

[project.scripts]
my-cli = "my_package.cli:main"

[project.entry-points."my_package.plugins"]
default = "my_package.plugins.default:DefaultPlugin"

# -- Build config --
[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.wheel]
packages = ["src/my_package"]

[tool.hatch.build.targets.sdist]
include = ["src/my_package", "tests"]

# -- Tool config --
[tool.ruff]
target-version = "py310"
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "TCH"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.mypy]
python_version = "3.10"
strict = true
```

## Project Layouts

### src Layout (recommended)

```
my-package/
├── pyproject.toml
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core.py
│       └── py.typed         # Marker for PEP 561 typed packages
└── tests/
    ├── __init__.py
    └── test_core.py
```

### Flat Layout (simpler, legacy)

```
my-package/
├── pyproject.toml
├── my_package/
│   ├── __init__.py
│   └── core.py
└── tests/
    └── test_core.py
```

**Use src layout** -- prevents accidentally importing the local package instead of the installed one during testing.

## Dependency Management

### Pinning Strategy

```toml
# In library pyproject.toml: use >= lower bounds, avoid upper caps
dependencies = [
    "httpx>=0.25",           # Lower bound only
    "pydantic>=2.0,<3.0",   # Cap only at known-breaking major versions
]

# In application: pin exact versions in a lock file
# Use: pip-compile, pdm lock, or uv lock
```

### Optional Dependencies (Extras)

```toml
[project.optional-dependencies]
# Feature extras
postgres = ["asyncpg>=0.29", "psycopg[binary]>=3.1"]
redis = ["redis>=5.0"]
all = ["my-package[postgres,redis]"]

# Dev extras
dev = ["my-package[all]", "pytest>=8.0", "ruff>=0.4"]
```

```bash
# Install with extras
pip install my-package[postgres]
pip install -e ".[dev]"
```

## Versioning

### Dynamic with hatch-vcs (Git tags)

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
dynamic = ["version"]

[tool.hatch.version]
source = "vcs"

# Version written to file at build time
[tool.hatch.version.raw-options]
local_scheme = "no-local-version"  # Required for PyPI
```

```bash
# Create a version by tagging
git tag v1.2.3
git push --tags
# Build will use 1.2.3 as the version
```

### Manual Versioning

```toml
[project]
version = "1.2.3"
```

```python
# src/my_package/__init__.py
__version__ = "1.2.3"

# Keep in sync -- or use importlib.metadata:
from importlib.metadata import version
__version__ = version("my-package")
```

## Editable Installs

```bash
# Modern editable install (PEP 660)
pip install -e .

# With extras
pip install -e ".[dev]"

# If using hatchling, editable installs use import hooks (fast, no .pth hacks)
# If using setuptools, may need: pip install -e . --config-settings editable_mode=compat
```

## Building and Publishing

### Build

```bash
# Install build tool
pip install build

# Build sdist and wheel
python -m build
# Output in dist/
#   my_package-1.2.3.tar.gz  (sdist)
#   my_package-1.2.3-py3-none-any.whl  (wheel)
```

### Publish to PyPI

```bash
# Traditional: twine
pip install twine
twine upload dist/*

# Check first (validates metadata)
twine check dist/*
```

### Trusted Publishers (GitHub Actions -- recommended)

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  release:
    types: [published]

permissions:
  id-token: write  # Required for trusted publishing

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi  # Must match PyPI trusted publisher config
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Needed for hatch-vcs

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Build
        run: |
          pip install build
          python -m build

      - name: Publish
        uses: pypa/gh-action-pypi-publish@release/v1
        # No password needed -- uses OIDC trusted publishing
```

**Setup on PyPI**: Go to project settings -> Publishing -> Add GitHub as trusted publisher. Specify repo, workflow file, and environment name.

### Test PyPI First

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ my-package
```

## conda-forge Recipe Basics

```yaml
# recipe/meta.yaml
{% set name = "my-package" %}
{% set version = "1.2.3" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz
  sha256: <sha256-of-sdist>

build:
  noarch: python
  number: 0
  script: {{ PYTHON }} -m pip install . -vv --no-deps --no-build-isolation

requirements:
  host:
    - python >=3.10
    - pip
    - hatchling
    - hatch-vcs
  run:
    - python >=3.10
    - httpx >=0.25
    - pydantic >=2.0

test:
  imports:
    - my_package
  requires:
    - pytest
  commands:
    - pytest tests/

about:
  home: https://github.com/user/my-package
  license: MIT
  license_file: LICENSE
  summary: One-line description
```

```bash
# Submit to conda-forge: fork conda-forge/staged-recipes, add recipe/, PR
# After merge, a feedstock repo is auto-created
# Updates: bump version in feedstock, bot often does this automatically
```

## Monorepo Packaging

```
monorepo/
├── pyproject.toml           # Workspace root (not a package)
├── packages/
│   ├── core/
│   │   ├── pyproject.toml   # name = "myorg-core"
│   │   └── src/myorg_core/
│   ├── api/
│   │   ├── pyproject.toml   # name = "myorg-api", depends on myorg-core
│   │   └── src/myorg_api/
│   └── worker/
│       ├── pyproject.toml   # name = "myorg-worker", depends on myorg-core
│       └── src/myorg_worker/
```

```toml
# packages/api/pyproject.toml
[project]
name = "myorg-api"
dependencies = [
    "myorg-core",  # Published as separate package
    "fastapi>=0.110",
]

# For local development, install all packages in editable mode:
# pip install -e packages/core -e packages/api -e packages/worker
```

**Tools**: uv workspaces, pdm workspaces, or plain pip with editable installs.

## Gotchas

- **Forgetting `py.typed` marker**: without `src/my_package/py.typed`, mypy won't use your type hints from installed package
- **`find_packages()` in src layout**: if using setuptools, must set `package_dir = {"": "src"}` and `packages = find_packages(where="src")`
- **Upper version caps on dependencies**: `httpx>=0.25,<1.0` breaks users when httpx 1.0 ships and is compatible; only cap at known-breaking versions
- **Missing `fetch-depth: 0` in CI**: hatch-vcs needs full git history to compute version from tags; shallow clones get `0.0.0`
- **`__init__.py` imports breaking install**: if `__init__.py` imports from dependencies, `pip install` fails because deps aren't installed yet during metadata extraction; use lazy imports
- **LICENSE file not included in wheel**: add `license-files = ["LICENSE"]` or ensure your backend includes it by default (hatchling does)
- **Namespace packages**: if you want `myorg.core` and `myorg.api` as separate packages, omit `__init__.py` in the `myorg/` directory; use implicit namespace packages (PEP 420)
- **Testing against installed package**: always `pip install -e .` before running tests; never test against local source without installing (import paths differ)
- **Forgetting to bump version**: CI should fail if you push a tag that doesn't match the package version; automate with hatch-vcs
