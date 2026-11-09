"""Compile a snippet and confirm every import resolves. Never executes the snippet.

usage: python check_snippet.py <file.py>
"""

import ast
import importlib
import importlib.metadata
import sys


def check_imports(tree: ast.AST) -> list[tuple[bool, str]]:
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append(_module(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            ok, line = _module(node.module)
            if not ok:
                results.append((ok, line))
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                results.append(_name(module, node.module, alias.name))
    return results


def _module(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, f"OK       import {name}"
    except Exception as e:  # noqa: BLE001 - report any import failure
        return False, f"MISSING  import {name}  ({type(e).__name__}: {e})"


def _name(module, module_name: str, name: str) -> tuple[bool, str]:
    if name == "*" or hasattr(module, name):
        return True, f"OK       from {module_name} import {name}"
    try:  # a submodule that is not imported by its package's __init__
        importlib.import_module(f"{module_name}.{name}")
        return True, f"OK       from {module_name} import {name}"
    except Exception:  # noqa: BLE001
        return False, f"MISSING  from {module_name} import {name}"


def versions() -> str:
    found = []
    for pkg in ("langchain", "langchain-core", "langgraph", "deepagents", "langsmith"):
        try:
            found.append(f"{pkg}=={importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return ", ".join(found)


def main(path: str) -> int:
    source = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(source, filename=path)
        compile(tree, path, "exec")
    except SyntaxError as e:
        print(f"SyntaxError: {e.msg} (line {e.lineno})")
        print("FAIL")
        return 1

    results = check_imports(tree)
    for _, line in results:
        print(line)
    print(f"checked against: {versions()}")
    failed = [line for ok, line in results if not ok]
    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
