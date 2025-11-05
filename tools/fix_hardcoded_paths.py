# Copyright (c) 2025 Vanargo
# Licensed under the MIT License. See LICENSE in the project root.

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nd_dir = ROOT / "notebooks"

ABS_PATTERNS = [
    r"[A-Za-z]:\\\\",  # Windows: C:\\
    r"[A-za-z]:/",  # Windows: C:/
    r"/Users/[^\s\"']+",  # macOs: /Users/...
    r"/home/[^\s\"']+",  # Linux: /home/...
    r"/mnt/[^\s\"']+",  # Linux: /mnt/...
]

abs_re = re.compile("|".join(ABS_PATTERNS))


def is_abs_present(s: str) -> bool:
    return bool(abs_re.search(s))


def load_ipynb(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save_ipynb(p: Path, obj: dict):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def replace_common_roots(text: str) -> str:
    """
    Простая эвристика: известные подпути проекта меняем на Path-выражения.
    Требует ручной проверки диффа.
    """
    replacements = {
        r"[A-Za-z]:\\[^\"\']*?1\. Census Income Classifier_v02\\data\\processed\\": "PROC_DIR / ",
        r"[A-Za-z]:\\[^\"\']*?1\. Census Income Classifier_v02\\data\\raw\\": "RAW_DIR / ",
        r"[A-Za-z]:\\[^\"\']*?1\. Census Income Classifier_v02\\data\\interim\\": "INT_DIR / ",
        r"[A-Za-z]:\\[^\"\']*?1\. Census Income Classifier_v02\\artifacts\\": "ART_DIR / ",
        r"[A-Za-z]:\\[^\"\']*?1\. Census Income Classifier_v02\\models\\": "MODELS_DIR / ",
        r"[A-Za-z]:\\[^\"\']*?1\. Census Income Classifier_v02\\reports\\": "REPORTS_DIR / ",
        r"[A-Za-z]:\\[^\"\']*?1\. Census Income Classifier_v02\\notebooks\\": "NB_DIR / ",
        r"/[^\"\']*/1\. Census Income Classifier_v02/data/processed/": "PROC_DIR / ",
        r"/[^\"\']*/1\. Census Income Classifier_v02/data/raw/": "RAW_DIR / ",
        r"/[^\"\']*/1\. Census Income Classifier_v02/data/interim/": "INT_DIR / ",
        r"/[^\"\']*/1\. Census Income Classifier_v02/artifacts/": "ART_DIR / ",
        r"/[^\"\']*/1\. Census Income Classifier_v02/models/": "MODELS_DIR / ",
        r"/[^\"\']*/1\. Census Income Classifier_v02/reports/": "REPORTS_DIR / ",
        r"/[^\"\']*/1\. Census Income Classifier_v02/notebooks/": "NB_DIR / ",
    }
    out = text
    for pat, repl in replacements.items():
        out = re.sub(pat, repl, out)
    return out


def process_notebook(nb_path: Path):
    nb = load_ipynb(nb_path)
    changed = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if is_abs_present(src):
            # сначала грубая подстановка известных подпапок
            new_src = replace_common_roots(src)
            # рекомендация приводить строковые пути к Path (вручную посмотреть дифф)
            if new_src != src:
                cell["source"] = new_src
                changed = True
    if changed:
        backup = nb_path.with_suffix(".backup.ipynb")
        backup.write_text(nb_path.read_text(encoding="utf-8"), encoding="utf-8")
        save_ipynb(nb_path, nb)
        print(f"[fixed] {nb_path} (backup -> {backup.name})")


if __name__ == "__main__":
    for p in (ROOT / "notebooks").rglob("*.ipynb"):
        process_notebook(p)
