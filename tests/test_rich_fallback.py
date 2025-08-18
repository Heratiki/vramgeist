import importlib
import sys
from src.vramgeist import _rich_fallback as rf


def test_simple_console_print(capsys):
    # Use fallback Console when rich is not installed (common in test env)
    Console = rf.Console
    c = Console()
    c.print('hello', '[red]danger[/red]', 123)
    out = capsys.readouterr().out
    assert 'hello' in out
    assert 'danger' in out
    assert '123' in out


def test_panel_and_box_str():
    Panel = rf.Panel
    box = rf.box
    p = Panel('content', title='t')
    assert str(p) == 'content'
    # Box should expose attributes like DOUBLE or ROUNDED
    assert hasattr(box, 'DOUBLE') or hasattr(box, 'ROUNDED')


def test_table_and_layout():
    Table = rf.Table
    Layout = rf.Layout
    t = Table()
    t.add_column('c')
    t.add_row('a')
    s = str(t)
    assert 'a' in s
    L = Layout()
    # split_column expects section-like objects; verify no exception
    class Sec:
        name = 'model_info'
    L.split_column(Sec())
    # indexing returns a panel or placeholder
    assert L['model_info'] is not None
