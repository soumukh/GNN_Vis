"""Interactive GNN Explainer: Dash entry point.

Run ``python app.py`` and open http://127.0.0.1:8050/.

This file only creates the Dash app. The code lives in the ``gnnvis`` package:

    gnnvis/models/          GNN models, registry, package loading
    gnnvis/runtime.py       device, caches of packages / datasets / models, locks
    gnnvis/ui/layout/       page layout, one module per area
    gnnvis/ui/callbacks/    callbacks, one module per feature (registered on import)
    gnnvis/ui/figures/      figure builders; theme.py / components.py: shared look
    gnnvis/config.py        datasets, defaults, model list

Analysis code: app_stability.py, app_expl.py, cpa_iv_link_prediction.py,
link_prediction_utils.py. Offline scripts: train_models.py, precompute_*.py.
"""

# torch must be imported before dash: on Windows, loading dash first makes
# torch's c10.dll fail to initialise (WinError 1114).
import torch  # noqa: F401

import os
import sys
import warnings

import dash

from app_stability import _scan_link_stability_cache
from gnnvis.ui.layout import build_layout

# Logs contain emoji/arrows: when stdout is redirected on Windows the default
# cp1252 encoding raises UnicodeEncodeError inside callbacks (e.g. aborting the
# model load). Force UTF-8 and never fail on a print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
sys.stdout.flush()
sys.stderr.flush()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress Dash callback graph warnings
warnings.filterwarnings("ignore", message="Can not create edge")

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
])
server = app.server
app.title = "🧠 Interactive GNN Explainer"
app.layout = build_layout()

# Registers every callback (dash.callback / dash.clientside_callback)
import gnnvis.ui.callbacks  # noqa: E402,F401


if __name__ == '__main__':
    print("Dash app starting...")
    _scan_link_stability_cache()
    # Disable the Werkzeug auto-reloader on Windows to avoid select() socket errors
    # (WinError 10038) caused by the reloader thread closing sockets unexpectedly.
    # Bind to localhost by default: debug mode exposes the Werkzeug debugger, which
    # allows code execution. Set GNN_VIS_HOST=0.0.0.0 / GNN_VIS_DEBUG=0 to change.
    _host = os.environ.get('GNN_VIS_HOST', '127.0.0.1')
    _debug = os.environ.get('GNN_VIS_DEBUG', '1') == '1'
    # dev_tools_hot_reload=False: with the code reloader off, hot reload only
    # made pages left open across a server restart reconnect and replay their
    # callbacks against the new server. Their browser-side (clientside)
    # functions are the old ones, so new clientside callbacks failed with
    # "Cannot read properties of undefined (reading 'apply')". Refresh instead.
    app.run(debug=_debug, host=_host, port=8050, use_reloader=False,
            dev_tools_hot_reload=False)
