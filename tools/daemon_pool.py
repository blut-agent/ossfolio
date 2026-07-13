"""Shared daemon-thread ThreadPoolExecutor.

Stdlib ``ThreadPoolExecutor`` workers are non-daemon AND are registered in
``concurrent.futures.thread._threads_queues``, whose atexit hook
(``_python_exit``) joins every worker unconditionally — even after
``shutdown(wait=False)``.  A single wedged worker (tool blocked on network
I/O, hung provider daemon, stuck subagent) therefore blocks interpreter
exit forever.  This is the root cause of multi-minute CLI exits on long
sessions: every abandoned concurrent-tool batch leaves workers that the
exit hook insists on joining.

``DaemonThreadPoolExecutor`` spawns daemon workers and skips the
``_threads_queues`` registration, so:

  - ``_python_exit`` never joins them, and
  - the interpreter's non-daemon thread join at shutdown skips them.

Semantics are otherwise identical (initializer/initargs, work queue,
idle-thread reuse).  Use it for any pool whose work is best-effort or
independently interruptible and must never hold the process open:
concurrent tool execution, background memory sync, catalog fan-out,
subagent timeout wrappers.  Do NOT use it for work that must complete
before exit (durable writes) — those belong on foreground threads with
explicit bounded joins.
"""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker

__all__ = ["DaemonThreadPoolExecutor"]


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def __init__(self, *args, **kwargs):
        # Python 3.14+ refactored ThreadPoolExecutor to store initializer/initargs
        # in a WorkerContext closure via prepare_context() instead of as
        # self._initializer / self._initargs. Our _adjust_thread_count() still
        # references those attributes directly (to keep the daemon-thread logic
        # close to the CPython 3.8-3.13 template).  Explicitly set them here so
        # the override works on both old and new stdlib versions.  We set them
        # BEFORE calling super() so the base class can also read them (it does
        # on Python 3.13 and earlier) and we also set them AFTER in case the
        # base class overwrites them (Python 3.14+).
        initializer = kwargs.get("initializer", None)
        initargs = kwargs.get("initargs", ())
        self._initializer = initializer
        self._initargs = initargs
        super().__init__(*args, **kwargs)
        # Python 3.14+ overwrites self._initializer / self._initargs in __init__
        # (via prepare_context). Re-assert our copies so _adjust_thread_count()
        # sees the right values.
        self._initializer = initializer
        self._initargs = initargs

    def _adjust_thread_count(self) -> None:
        # Mirrors CPython's implementation (3.8–3.13) with two changes:
        # daemon=True and no _threads_queues registration.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            t.start()
            self._threads.add(t)
