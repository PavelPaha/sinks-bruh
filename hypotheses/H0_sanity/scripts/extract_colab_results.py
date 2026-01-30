"""
Автоматически находит и распаковывает h0_results.zip из Colab в правильную директорию.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def find_zip_file() -> Path | None:
    """Ищет h0_results.zip в стандартных местах."""
    # Проверяем текущую директорию
    cwd = Path.cwd()
    if (cwd / "h0_results.zip").exists():
        return cwd / "h0_results.zip"
    
    # Проверяем Downloads
    downloads = Path.home() / "Downloads"
    if (downloads / "h0_results.zip").exists():
        return downloads / "h0_results.zip"
    
    # Проверяем Desktop
    desktop = Path.home() / "Desktop"
    if (desktop / "h0_results.zip").exists():
        return desktop / "h0_results.zip"
    
    return None


def main():
    script_path = Path(__file__).resolve()
    hyp_dir = script_path.parents[1]  # hypotheses/H0_sanity
    data_dir = hyp_dir / "data"
    
    print("Looking for h0_results.zip...")
    zip_path = find_zip_file()
    
    if not zip_path:
        print("❌ h0_results.zip not found in:")
        print("   - Current directory")
        print("   - ~/Downloads")
        print("   - ~/Desktop")
        print("\nPlease:")
        print("  1. Download h0_results.zip from Colab")
        print("  2. Place it in one of the above locations")
        print("  3. Re-run this script")
        return
    
    print(f"✅ Found: {zip_path}")
    
    # Создаём временную директорию для распаковки
    temp_dir = data_dir.parent / "temp_extract"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"📦 Extracting to {temp_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Копируем файлы в data/
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = list(temp_dir.rglob("*"))
    
    copied = 0
    for f in extracted_files:
        if f.is_file() and f.suffix in [".gz", ".json"]:
            dest = data_dir / f.name
            shutil.copy2(f, dest)
            print(f"  ✓ Copied {f.name} ({dest.stat().st_size} bytes)")
            copied += 1
    
    # Удаляем временную директорию
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ Extracted {copied} file(s) to {data_dir}")
    print(f"\nNext: run 'python hypotheses/H0_sanity/scripts/check_structure.py'")


if __name__ == "__main__":
    main()
