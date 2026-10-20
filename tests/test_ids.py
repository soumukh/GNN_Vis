"""Callbacks and layout agree on component ids.

The app runs with suppress_callback_exceptions=True, so Dash does not report a
callback whose component does not exist: it simply never fires. These tests do.
"""
import ast
import glob
import json
import os

import plotly

from gnnvis.ui import ids
from gnnvis.ui.layout import build_layout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID_VALUES = {k: v for k, v in vars(ids).items() if k.isupper()}


def _layout_ids():
    data = json.loads(json.dumps(build_layout(), cls=plotly.utils.PlotlyJSONEncoder))
    found = set()

    def walk(c):
        if isinstance(c, dict) and 'props' in c:
            if isinstance(c['props'].get('id'), str):
                found.add(c['props']['id'])
            for v in c['props'].values():
                walk(v)
        elif isinstance(c, list):
            for x in c:
                walk(x)
    walk(data)
    return found


def _dynamic_ids():
    """String ids of components that callbacks / builders create at run time
    (e.g. the selection inputs, the cell-details table and its pager)."""
    found = set()
    for f in glob.glob(os.path.join(ROOT, 'gnnvis', 'ui', '**', '*.py'), recursive=True):
        if os.sep + 'layout' + os.sep in f:
            continue
        for n in ast.walk(ast.parse(open(f, encoding='utf-8').read())):
            for k in getattr(n, 'keywords', []):
                if k.arg == 'id' and isinstance(k.value, ast.Attribute) and ast.unparse(k.value.value) == 'ids':
                    found.add(ID_VALUES[k.value.attr])
    return found


def _callback_ids():
    import dash._callback as dcb
    import gnnvis.ui.callbacks  # noqa: F401  (registers the callbacks)
    used = set()
    for cb in dcb.GLOBAL_CALLBACK_LIST:
        out = cb['output']
        parts = out[2:-2].split('...') if out.startswith('..') else [out]
        for p in parts:
            cid = p.split('@')[0].rsplit('.', 1)[0]
            if not cid.startswith('{'):
                used.add(cid)
        for dep in cb['inputs'] + cb['state']:
            # pattern-matching ids are stored as JSON strings ('{"index": ..., "type": ...}')
            if isinstance(dep['id'], str) and not dep['id'].startswith('{'):
                used.add(dep['id'])
    return used


def test_every_callback_id_exists():
    available = _layout_ids() | _dynamic_ids()
    missing = sorted(_callback_ids() - available)
    assert missing == [], f"callbacks reference components that are never created: {missing}"


def test_every_id_constant_is_used():
    sources = ''.join(open(f, encoding='utf-8').read()
                      for f in glob.glob(os.path.join(ROOT, 'gnnvis', 'ui', '**', '*.py'), recursive=True)
                      if not f.endswith('ids.py'))
    unused = sorted(k for k in ID_VALUES if f'ids.{k}' not in sources)
    assert unused == []


def test_id_values_are_unique():
    assert len(set(ID_VALUES.values())) == len(ID_VALUES)
