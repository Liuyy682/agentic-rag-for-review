import re
from pathlib import Path


def test_application_package_does_not_mutate_sys_path():
    package_root = Path(__file__).resolve().parents[2] / "src" / "agentic_rag"
    offenders = [
        str(path)
        for path in package_root.rglob("*.py")
        if "sys.path.insert" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
def test_application_package_uses_namespaced_imports():
    package_root = Path(__file__).resolve().parents[2] / "src" / "agentic_rag"
    legacy_import = re.compile(
        r"^\s*(?:from|import)\s+(?:api|application|chat|config|core|ingestion|observability|rag_agent|retrieval|storage)(?:\.|\s|$)",
        re.MULTILINE,
    )
    offenders = [
        str(path)
        for path in package_root.rglob("*.py")
        if legacy_import.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
