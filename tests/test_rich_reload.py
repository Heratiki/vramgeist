import importlib
import sys
import types


def test_reload_rich_fallback_with_no_rich(monkeypatch):
    # Ensure 'rich' modules not present
    monkeypatch.setitem(sys.modules, 'rich', None)
    for sub in ['rich.console', 'rich.panel', 'rich.table', 'rich.layout', 'rich.text', 'rich.align']:
        if sub in sys.modules:
            del sys.modules[sub]
    # Import the module fresh
    if 'src.vramgeist._rich_fallback' in sys.modules:
        del sys.modules['src.vramgeist._rich_fallback']
    rf = importlib.import_module('src.vramgeist._rich_fallback')
    # Expect fallback Console class
    Console = rf.Console
    c = Console()
    c.print('x')
    # SimpleTable should be fallback
    Table = rf.Table
    t = Table()
    t.add_row('a')
    assert 'a' in str(t)


def test_reload_rich_fallback_with_fake_rich(monkeypatch):
    # Create fake rich modules with minimal attrs
    mod_console = types.SimpleNamespace(Console=lambda *a, **k: 'console')
    mod_panel = types.SimpleNamespace(Panel=lambda *a, **k: 'panel')
    mod_table = types.SimpleNamespace(Table=lambda *a, **k: types.SimpleNamespace(add_row=lambda *r: None, __str__=lambda self: ''))
    mod_layout = types.SimpleNamespace(Layout=lambda *a, **k: types.SimpleNamespace(split_column=lambda *s: None, __getitem__=lambda self, k: None))
    mod_text = types.SimpleNamespace(Text=str)
    mod_align = types.SimpleNamespace(Align=types.SimpleNamespace(center=lambda x: x))

    monkeypatch.setitem(sys.modules, 'rich.console', mod_console)
    monkeypatch.setitem(sys.modules, 'rich.panel', mod_panel)
    monkeypatch.setitem(sys.modules, 'rich.table', mod_table)
    monkeypatch.setitem(sys.modules, 'rich.layout', mod_layout)
    monkeypatch.setitem(sys.modules, 'rich.text', mod_text)
    monkeypatch.setitem(sys.modules, 'rich.align', mod_align)
    # Also put a rich module with 'box' attribute
    rich_mod = types.SimpleNamespace(box=types.SimpleNamespace(ROUNDED='r', DOUBLE='d'))
    monkeypatch.setitem(sys.modules, 'rich', rich_mod)

    if 'src.vramgeist._rich_fallback' in sys.modules:
        del sys.modules['src.vramgeist._rich_fallback']
    rf = importlib.import_module('src.vramgeist._rich_fallback')
    # Now Console should be from fake rich
    assert rf._Console is not None
    assert rf.Console is not None
    # Panel should be from fake rich
    assert rf._Panel is not None
    assert rf.Panel is not None