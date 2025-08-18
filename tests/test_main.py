import sys
import types
import runpy
import importlib


def test_import_does_not_call_main(monkeypatch):
    """Importing `vramgeist.__main__` should not call `cli.main` (only running as __main__ should)."""
    called = {"val": False}

    def fake_main():
        called["val"] = True

    fake_cli = types.ModuleType("vramgeist.cli")
    fake_cli.main = fake_main
    # Ensure the import inside __main__ picks up our fake module
    monkeypatch.setitem(sys.modules, "vramgeist.cli", fake_cli)

    # Force a fresh import of the __main__ module
    sys.modules.pop("vramgeist.__main__", None)
    importlib.import_module("vramgeist.__main__")

    assert called["val"] is False


def test_run_module_calls_main(monkeypatch):
    """Running the package as a module (python -m vramgeist) should call `cli.main`."""
    called = {"count": 0}

    def fake_main():
        called["count"] += 1

    fake_cli = types.ModuleType("vramgeist.cli")
    fake_cli.main = fake_main
    monkeypatch.setitem(sys.modules, "vramgeist.cli", fake_cli)

    # Ensure we start with a clean __main__ module
    sys.modules.pop("vramgeist.__main__", None)
    runpy.run_module("vramgeist", run_name="__main__")

    assert called["count"] > 0
