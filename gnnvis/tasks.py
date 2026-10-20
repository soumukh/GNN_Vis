"""The analysis tasks the app supports and what differs between them.

The task switch stores a task key (``analysis-mode-store``); callbacks look the
task up with ``get_task(key)`` instead of comparing strings. Explainers are
registered per task (gnnvis/explainers): ``task.explainer(kind)``.
"""

from dataclasses import dataclass

NODE_CLASSIFICATION = 'node_classification'
LINK_PREDICTION = 'link_prediction'


@dataclass(frozen=True)
class Task:
    key: str
    label: str                 # status bar
    switch_label: str          # task switch in the header
    graph_type: str            # 'full' graph or a local 'sub'graph
    confusion_min_height: str  # 7x7 node matrix needs more room than 2x2 links
    cell_details_hint: str     # cell-details panel before a cell is clicked
    idle_hint: str             # status bar, nothing selected yet
    detail_empty_hint: str     # Explanation tab, nothing to show yet

    @property
    def is_link(self):
        """True if the task's targets are (source, target) pairs, not nodes."""
        return self.key == LINK_PREDICTION

    def explainer(self, kind, name=None):
        from gnnvis.explainers import get_explainer
        return get_explainer(self.key, kind, name)


NODE = Task(
    key=NODE_CLASSIFICATION,
    label='Node classification',
    switch_label=' Node',
    graph_type='full',
    confusion_min_height='380px',
    cell_details_hint='the nodes in it; click a row to select that node',
    idle_hint='Click a node in the embedding plot or full graph, or type an ID under Selection.',
    detail_empty_hint='Load a model to begin.',
)

LINK = Task(
    key=LINK_PREDICTION,
    label='Link prediction',
    switch_label=' Link',
    graph_type='sub',
    confusion_min_height='240px',
    cell_details_hint='its edges (TP / FP / FN / TN); click a row to inspect that link',
    idle_hint=('Enter source and target node IDs and press RUN ANALYSIS, or click a '
               'confusion-matrix cell and then a link in its list.'),
    detail_empty_hint='Pick a predicted link from the list above to inspect it.',
)

TASKS = {t.key: t for t in (NODE, LINK)}
DEFAULT_TASK = NODE_CLASSIFICATION


def get_task(key):
    """The task for a stored key (the default task for None / unknown keys)."""
    return TASKS.get(key) or TASKS[DEFAULT_TASK]
