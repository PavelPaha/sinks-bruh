#!/bin/bash
# Копирует файлы из Downloads в правильную директорию

SOURCE_DIR="$HOME/Downloads"
TARGET_DIR="$(dirname "$0")/../data"

echo "Looking for H0 results in $SOURCE_DIR..."

# Ищем файлы
FILES=(
    "mmlu__Qwen__Qwen2.5-0.5B-Instruct__K4__qlast.jsonl.gz"
    "mmlu__Qwen__Qwen2.5-0.5B-Instruct__K4__qlast.meta.json"
)

mkdir -p "$TARGET_DIR"
copied=0

for file in "${FILES[@]}"; do
    src="$SOURCE_DIR/$file"
    dst="$TARGET_DIR/$file"
    
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        size=$(stat -f%z "$dst" 2>/dev/null || stat -c%s "$dst" 2>/dev/null || echo "?")
        echo "  ✓ Copied $file ($size bytes)"
        copied=$((copied + 1))
    else
        echo "  ⚠️  Not found: $file"
    fi
done

if [ $copied -eq 0 ]; then
    echo ""
    echo "❌ No files found in $SOURCE_DIR"
    echo ""
    echo "Please:"
    echo "  1. Download files from Colab (run cell 8 in the notebook)"
    echo "  2. Files should appear in ~/Downloads"
    echo "  3. Re-run this script"
else
    echo ""
    echo "✅ Copied $copied file(s) to $TARGET_DIR"
    echo ""
    echo "Next: run 'python hypotheses/H0_sanity/scripts/check_structure.py'"
fi
