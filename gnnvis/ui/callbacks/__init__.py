"""Dash callbacks, one module per feature. Importing this package registers them all."""

from gnnvis.ui.callbacks import session  # noqa: F401
from gnnvis.ui.callbacks import selection  # noqa: F401
from gnnvis.ui.callbacks import explanation  # noqa: F401
from gnnvis.ui.callbacks import overview  # noqa: F401
from gnnvis.ui.callbacks import link  # noqa: F401
from gnnvis.ui.callbacks import full_graph  # noqa: F401
from gnnvis.ui.callbacks import stability  # noqa: F401
