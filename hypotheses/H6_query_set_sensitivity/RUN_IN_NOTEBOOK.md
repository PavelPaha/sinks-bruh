## Notebook запуск

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
subprocess.check_call([sys.executable, "hypotheses/H6_query_set_sensitivity/scripts/run_all.py", "--measure"])
```

### 2) Построить графики/метрики + написать `final.md`

```python
import sys, subprocess
subprocess.check_call([sys.executable, "hypotheses/H6_query_set_sensitivity/scripts/run_all.py"])
```

Где смотреть результаты:
- `hypotheses/H6_query_set_sensitivity/data/` — сырые `.jsonl.gz`
- `hypotheses/H6_query_set_sensitivity/plots/` — метрики + графики
- `hypotheses/H6_query_set_sensitivity/final.md` — авто-текст (под статью)
