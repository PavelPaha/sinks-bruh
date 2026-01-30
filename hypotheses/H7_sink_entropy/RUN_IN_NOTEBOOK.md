## Run H7 in a notebook (cloud-friendly)

### Cell 0 — install deps (once per kernel)

```python
import sys, subprocess
from pathlib import Path

repo = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / "pyproject.toml").exists())
req = repo / "requirements-hypotheses.txt"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
```

### Cell 1 — run measurements

```python
import os, sys, subprocess
os.chdir(str(repo))

subprocess.check_call([sys.executable, "hypotheses/H7_sink_entropy/scripts/run_all.py", "--measure"])
```

### Cell 2 — build plots + summary

```python
import os, sys, subprocess
os.chdir(str(repo))

subprocess.check_call([sys.executable, "hypotheses/H7_sink_entropy/scripts/run_all.py"])
```

### Cell 3 — extra aggregated plots (recommended)

```python
import os, sys, subprocess
os.chdir(str(repo))

subprocess.check_call([sys.executable, "hypotheses/H7_sink_entropy/scripts/aggregate_analysis.py", "--metric", "spearman"])
subprocess.check_call([sys.executable, "hypotheses/H7_sink_entropy/scripts/aggregate_analysis.py", "--metric", "partial_seq_len"])
subprocess.check_call([sys.executable, "hypotheses/H7_sink_entropy/scripts/aggregate_analysis.py", "--metric", "pearson"])
```

