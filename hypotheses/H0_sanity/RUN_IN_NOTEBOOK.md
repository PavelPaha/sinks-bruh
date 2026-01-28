## Notebook запуск (H0)

### 0) Установить зависимости (один раз на окружение)

```python
import sys, subprocess
from pathlib import Path

# Find repo root (where pyproject.toml lives), regardless of current CWD
repo = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / "pyproject.toml").exists())
req = repo / "requirements-hypotheses.txt"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
```

### 1) (опционально) Запустить измерение

```python
import sys, subprocess
subprocess.check_call([sys.executable, "hypotheses/H0_sanity/scripts/run_all.py", "--measure"])
```

Это возьмёт `hypotheses/H0_sanity/config.json`, прогонит `scripts/measure_sink_text.py` и положит `.jsonl.gz` в `hypotheses/H0_sanity/data/`.

### 2) Построить графики + написать `final.md`

```python
import sys, subprocess
subprocess.check_call([sys.executable, "hypotheses/H0_sanity/scripts/run_all.py"])
```

Артефакты:
- `hypotheses/H0_sanity/plots/` (+ `summary.json`, `manifest.json`)
- `hypotheses/H0_sanity/final.md`

