import ast
import os
import sys
import importlib.util
from pathlib import Path

def find_python_files(root_dir):
    return [f for f in Path(root_dir).rglob("*.py") if f.is_file()]

def extract_imports(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            node = ast.parse(f.read(), filename=str(file_path))
    except SyntaxError:
        return set()
    
    imports = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module and not n.level:
                imports.add(n.module.split('.')[0])
    return imports

def is_third_party(module):
    if module in sys.builtin_module_names:
        return False
    try:
        spec = importlib.util.find_spec(module)
        if spec and spec.origin:
            path = spec.origin
            return "site-packages" in path or "dist-packages" in path
        return False
    except Exception:
        return False

def main(root_dir="."):
    all_imports = set()
    for file in find_python_files(root_dir):
        all_imports |= extract_imports(file)

    third_party = sorted({imp for imp in all_imports if is_third_party(imp)})

    with open("requirements.txt", "w") as f:
        for pkg in third_party:
            f.write(f"{pkg}\n")
    
    print(f"[+] requirements.txt written with {len(third_party)} third-party packages.")

if __name__ == "__main__":
    main()
