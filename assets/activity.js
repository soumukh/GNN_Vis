/*
 * Global activity indicator (auto-loaded by Dash from assets/).
 *
 * Dash marks components whose callbacks are in flight with
 * data-dash-is-loading="true". This polls for those markers and shows a thin
 * animated bar plus a badge naming the slowest step in progress and the
 * elapsed time. It appears only after a short delay so quick updates do not
 * flicker, and it also disables "Run Stability Analysis" while that runs.
 */
(function () {
    // Highest priority first: the first match names the activity.
    var LABELS = [
        ['stability-status', 'Running stability analysis'],
        ['global-run-button', 'Running analysis (GNNExplainer)'],
        ['gnn-causal-explanation-view', 'Computing causal paths (CPA-IV)'],
        ['full-graph-view', 'Laying out full graph'],
        ['main-stability-plot', 'Drawing stability plots'],
        ['drilldown-scatter', 'Drawing stability plots'],
        ['graph-view', 'Building explanation views'],
        ['feature-importance-view', 'Building explanation views'],
        ['neighbor-importance-view', 'Building explanation views'],
        ['subgraph-legend-inline', 'Building explanation views'],
        ['edge-predictions-display', 'Listing cell contents'],
        ['embeddings-view', 'Updating overview'],
        ['confusion-matrix-plot', 'Updating overview'],
        ['performance-metrics', 'Updating overview'],
    ];
    var SHOW_AFTER_MS = 600;   // ignore quick updates
    var HIDE_AFTER_MS = 400;   // bridge short gaps between chained callbacks

    var bar, badge, labelEl, timeEl;
    var busySince = null, idleSince = null, visible = false;

    function build() {
        bar = document.createElement('div');
        bar.className = 'activity-bar';
        badge = document.createElement('div');
        badge.className = 'activity-badge';
        badge.setAttribute('role', 'status');
        badge.setAttribute('aria-live', 'polite');
        var spinner = document.createElement('span');
        spinner.className = 'activity-spinner';
        labelEl = document.createElement('span');
        timeEl = document.createElement('span');
        timeEl.className = 'activity-time';
        badge.appendChild(spinner);
        badge.appendChild(labelEl);
        badge.appendChild(timeEl);
        document.body.appendChild(bar);
        document.body.appendChild(badge);
    }

    function currentLabel(loadingIds) {
        var status = document.getElementById('status-bar');
        if (status && status.innerText.indexOf('Loading model') !== -1) {
            return 'Loading model';
        }
        for (var i = 0; i < LABELS.length; i++) {
            if (loadingIds.indexOf(LABELS[i][0]) !== -1) return LABELS[i][1];
        }
        return 'Working';
    }

    function loadingIds() {
        var nodes = document.querySelectorAll('[data-dash-is-loading="true"]');
        var ids = [];
        for (var i = 0; i < nodes.length; i++) {
            var id = nodes[i].id || '';
            if (!id) {
                var inner = nodes[i].querySelector('[id]');
                id = inner ? inner.id : '';
            }
            // the status line itself re-renders constantly; it is not "work"
            if (id && id !== 'status-bar') ids.push(id);
        }
        return ids;
    }

    function setStabilityButton(busy) {
        var btn = document.getElementById('run-stability-button');
        if (!btn) return;
        if (busy && !btn.disabled) { btn.disabled = true; btn.classList.add('is-busy'); }
        if (!busy && btn.classList.contains('is-busy')) { btn.disabled = false; btn.classList.remove('is-busy'); }
    }

    function tick() {
        if (!bar) build();
        var ids = loadingIds();
        var now = Date.now();
        setStabilityButton(ids.indexOf('stability-status') !== -1);

        if (ids.length) {
            idleSince = null;
            if (busySince === null) busySince = now;
            if (!visible && now - busySince >= SHOW_AFTER_MS) {
                visible = true;
                document.body.classList.add('is-busy');
            }
            if (visible) {
                labelEl.textContent = currentLabel(ids) + '…';
                timeEl.textContent = Math.floor((now - busySince) / 1000) + ' s';
            }
        } else {
            if (idleSince === null) idleSince = now;
            if (now - idleSince >= HIDE_AFTER_MS) {
                busySince = null;
                if (visible) {
                    visible = false;
                    document.body.classList.remove('is-busy');
                }
            }
        }
    }

    function start() { setInterval(tick, 200); }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
