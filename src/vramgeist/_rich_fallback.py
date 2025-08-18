"""Lightweight compatibility layer that provides Console, Panel and box
objects when 'rich' is not installed. If 'rich' is available we re-export
the real implementations, otherwise we provide minimal stand-ins so the
package remains importable in constrained/static-analysis environments.
"""
from types import SimpleNamespace
import importlib

# Use importlib to avoid static analyzers flagging missing 'rich' package
try:
    _console_mod = importlib.import_module('rich.console')
    _panel_mod = importlib.import_module('rich.panel')
    _box_mod = importlib.import_module('rich')
    _Console = getattr(_console_mod, 'Console')
    _Panel = getattr(_panel_mod, 'Panel')
    _box = getattr(_box_mod, 'box')
except Exception:
    _Console = None
    _Panel = None
    _box = None


class _SimpleConsole:
    def __init__(self, *args, **kwargs):
        pass

    def print(self, *args, **kwargs):
        # Simple fallback that joins args and prints to stdout
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        out = sep.join(str(a) for a in args)
        # strip some markup-like tokens so output is readable
        out = out.replace('[red]', '').replace('[/red]', '')
        print(out, end=end)


class _SimplePanel:
    def __init__(self, content, **kwargs):
        self.content = content
        self.kwargs = kwargs

    def __str__(self):
        return str(self.content)


class _SimpleBox(SimpleNamespace):
    ROUNDED = 'ROUNDED'
    DOUBLE = 'DOUBLE'


class _SimpleTable:
    def __init__(self, *args, **kwargs):
        self.rows = []
        self.columns = []

    def add_column(self, *args, **kwargs):
        # columns are informational in this fallback
        self.columns.append((args, kwargs))

    def add_row(self, *cols):
        self.rows.append(tuple(cols))

    def __str__(self):
        return '\n'.join(' | '.join(map(str, r)) for r in self.rows)


class _SimpleLayout:
    def __init__(self, *args, **kwargs):
        # Simple layout stores named children
        self._children = {}

    def split_column(self, *sections):
        # sections are Layout(...) instances; store them by name if provided
        for sec in sections:
            if hasattr(sec, 'name'):
                self._children[sec.name] = sec

    def __getitem__(self, key):
        # Return a simple section object with update
        return self._children.get(key, _SimplePanel(''))

    def update(self, panel):
        # No-op in fallback
        pass


class _SimpleText(str):
    def __new__(cls, content, style=None):
        obj = str.__new__(cls, content)
        return obj

    def __add__(self, other):
        return _SimpleText(str(self) + str(other))


class _SimpleAlign:
    LEFT = 'left'
    CENTER = 'center'
    RIGHT = 'right'

    @classmethod
    def center(cls, content):
        # Return content as-is in fallback
        return content


# Export Table, Layout, Text, Align
Table = getattr(importlib.import_module('rich.table'), 'Table') if _Panel and _Console else _SimpleTable
Layout = getattr(importlib.import_module('rich.layout'), 'Layout') if _Panel and _Console else _SimpleLayout
Text = getattr(importlib.import_module('rich.text'), 'Text') if _Panel and _Console else _SimpleText
Align = getattr(importlib.import_module('rich.align'), 'Align') if _Panel and _Console else _SimpleAlign


# Export either the real rich objects or our fallbacks
Console = _Console or _SimpleConsole
Panel = _Panel or _SimplePanel
box = _box or _SimpleBox()
