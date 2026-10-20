"""Background jobs for long analyses, with progress and cancellation.

A callback submits the work and returns at once; the page polls the job
(a dcc.Interval) and shows its progress. The work function gets a
``progress(done, total)`` callable, which raises JobCancelled once the job is
cancelled, so a run stops at its next step.
"""

import threading
import time
import traceback
import uuid

RUNNING, DONE, FAILED, CANCELLED = 'running', 'done', 'failed', 'cancelled'


class JobCancelled(Exception):
    """Raised inside the job's thread by progress() after cancel()."""


class Job:
    def __init__(self, label, unit):
        self.id = uuid.uuid4().hex[:12]
        self.label = label          # e.g. "Node stability, Cora / GCN, σ=0.05"
        self.unit = unit            # what progress counts: 'nodes', 'edges'
        self.done = 0
        self.total = None
        self.state = RUNNING
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None
        self._cancel = threading.Event()

    def progress(self, done, total=None):
        self.done = int(done)
        if total is not None:
            self.total = int(total)
        if self._cancel.is_set():
            raise JobCancelled()

    def cancel(self):
        self._cancel.set()

    @property
    def cancel_requested(self):
        return self._cancel.is_set()

    def snapshot(self):
        """Progress as plain data (for the UI)."""
        end = self.finished or time.time()
        elapsed = end - self.started
        eta = None
        if self.state == RUNNING and self.total and self.done:
            eta = elapsed / self.done * (self.total - self.done)
        return {'id': self.id, 'label': self.label, 'unit': self.unit, 'state': self.state,
                'done': self.done, 'total': self.total, 'elapsed': elapsed, 'eta': eta,
                'error': self.error, 'cancel_requested': self.cancel_requested}


class JobManager:
    """Runs jobs in daemon threads and keeps them until their result is taken
    (or, unclaimed, until more than ``keep`` jobs have finished)."""

    def __init__(self, keep=20):
        self._jobs = {}
        self._lock = threading.Lock()
        self._keep = keep

    def submit(self, label, unit, fn, *args, **kwargs):
        """Start ``fn(*args, progress=job.progress, **kwargs)``; returns the Job."""
        job = Job(label, unit)

        def run():
            try:
                job.result = fn(*args, progress=job.progress, **kwargs)
                job.state = DONE
            except JobCancelled:
                job.state = CANCELLED
            except Exception as e:
                traceback.print_exc()
                job.error = f"{type(e).__name__}: {e}"
                job.state = FAILED
            finally:
                job.finished = time.time()

        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        threading.Thread(target=run, name=f'job-{job.id}', daemon=True).start()
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id):
        job = self.get(job_id)
        if job is not None:
            job.cancel()
        return job

    def take(self, job_id):
        """Remove a finished job and return it (None if unknown or still running)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state == RUNNING:
                return None
            return self._jobs.pop(job_id)

    def running(self):
        with self._lock:
            return [j for j in self._jobs.values() if j.state == RUNNING]

    def _prune(self):
        finished = sorted((j for j in self._jobs.values() if j.state != RUNNING), key=lambda j: j.finished)
        for job in finished[:max(0, len(finished) - self._keep)]:
            del self._jobs[job.id]


JOBS = JobManager()
