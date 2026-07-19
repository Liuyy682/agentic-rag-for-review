from pathlib import Path


def test_application_package_does_not_mutate_sys_path():
    package_root = Path(__file__).resolve().parents[2] / "src" / "agentic_rag"
    offenders = [
        str(path)
        for path in package_root.rglob("*.py")
        if "sys.path.insert" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
